"""The single-stock verdict: is the business sound, is the price depressed, and is the drop temporary or structural?

Every rule reads one KPI row of the equity frame (plus the stock's sector
row for context) and records *why* it fired, so the answer can be audited.
Thresholds are defaults from the plan and live in :class:`VerdictThresholds`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Label = Literal[
    "quality_on_sale",
    "falling_knife",
    "sound_not_cheap",
    "oversold_unclear",
    "avoid",
    "insufficient_data",
]
DrawdownNature = Literal["idiosyncratic", "sector", "market", "none", "unknown"]


class VerdictThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_roic: float = 0.10
    min_fcf_conversion: float = 0.8
    max_accruals: float = 0.10
    max_net_debt_to_ebitda: float = 3.0
    min_interest_coverage: float = 4.0
    min_altman_z: float = 2.6
    max_dilution: float = 0.03
    min_piotroski: float = 6.0
    min_profitable_years: float = 3.0
    oversold_drawdown: float = -0.20
    oversold_rsi: float = 35.0
    idiosyncratic_excess: float = -0.10
    sound_score: float = 60.0
    oversold_score: float = 50.0


class Check(BaseModel):
    name: str
    passed: bool | None = Field(description="None when the input is missing")
    value: float | None
    threshold: str
    detail: str


class Evidence(BaseModel):
    signal: str
    reads_as: Literal["temporary", "structural", "neutral"]
    value: float | None
    detail: str


class StockVerdict(BaseModel):
    symbol: str
    as_of: str
    label: Label
    fundamentally_sound: bool | None
    fundamental_score: float | None = Field(
        description="Share of fundamental checks passed, 0-100"
    )
    fundamental_checks: list[Check]
    oversold: bool | None
    oversold_score: float | None = Field(
        description="Share of oversold signals firing, 0-100"
    )
    oversold_signals: list[Check]
    drawdown_nature: DrawdownNature
    likely_cause: str
    temporary_vs_structural: float | None = Field(
        description="+100 all temporary ... -100 all structural"
    )
    structural_red_flags: list[str]
    recovery_evidence: list[str]
    evidence: list[Evidence]
    data_coverage_grade: str | None = None
    summary: str


def _get(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _check(row: pd.Series, name: str, op: str, threshold: float, detail: str) -> Check:
    value = _get(row, name)
    passed = None
    if value is not None:
        passed = {
            ">=": value >= threshold,
            "<=": value <= threshold,
            ">": value > threshold,
            "<": value < threshold,
        }[op]
    return Check(
        name=name,
        passed=passed,
        value=value,
        threshold=f"{op} {threshold:g}",
        detail=detail,
    )


def fundamental_checks(row: pd.Series, t: VerdictThresholds) -> list[Check]:
    return [
        _check(
            row,
            "profitable_years_4",
            ">=",
            t.min_profitable_years,
            "Profitable in most of the last four fiscal years",
        ),
        _check(
            row,
            "fcf_positive_years_4",
            ">=",
            t.min_profitable_years,
            "Free cash flow positive in most of the last four years",
        ),
        _check(
            row,
            "roic",
            ">=",
            t.min_roic,
            "Return on invested capital above the cost of capital",
        ),
        _check(
            row,
            "fcf_conversion",
            ">=",
            t.min_fcf_conversion,
            "Earnings convert into cash",
        ),
        _check(
            row,
            "accruals_ratio",
            "<=",
            t.max_accruals,
            "Low accruals: earnings are not paper profits",
        ),
        _check(
            row,
            "revenue_decelerating_2y",
            "<",
            1.0,
            "Revenue growth not falling two years in a row",
        ),
        _check(row, "gross_margin_trend", ">=", -0.02, "Gross margin not eroding"),
        _check(
            row,
            "net_debt_to_ebitda",
            "<=",
            t.max_net_debt_to_ebitda,
            "Leverage manageable",
        ),
        _check(
            row,
            "interest_coverage",
            ">=",
            t.min_interest_coverage,
            "Interest comfortably covered",
        ),
        _check(row, "altman_z", ">=", t.min_altman_z, "Away from the distress zone"),
        _check(
            row, "share_count_change_1y", "<=", t.max_dilution, "No material dilution"
        ),
        _check(row, "piotroski_f", ">=", t.min_piotroski, "Piotroski F-score strong"),
    ]


def oversold_signals(row: pd.Series, t: VerdictThresholds) -> list[Check]:
    return [
        _check(
            row,
            "drawdown_52w",
            "<=",
            t.oversold_drawdown,
            "Well below the 52-week high",
        ),
        _check(row, "rsi_14", "<=", t.oversold_rsi, "RSI in oversold territory"),
        _check(
            row,
            "price_vs_sma200",
            "<",
            -0.10,
            "More than 10 % below the 200-day average",
        ),
        _check(
            row, "bollinger_pct_b", "<=", 0.1, "At or below the lower Bollinger band"
        ),
        _check(
            row,
            "ret_1m",
            "<=",
            -0.08,
            "Bottom of the one-month return range (reversal setup)",
        ),
        _check(
            row, "capitulation_volume", ">=", 2.0, "Capitulation volume on a down day"
        ),
        _check(
            row,
            "excess_drawdown_vs_sector",
            "<=",
            t.idiosyncratic_excess,
            "Fell much more than the sector",
        ),
    ]


def _score(checks: list[Check]) -> float | None:
    known = [c.passed for c in checks if c.passed is not None]
    return round(100 * sum(known) / len(known), 1) if known else None


def _nature(
    row: pd.Series, sector_row: pd.Series | None, t: VerdictThresholds
) -> DrawdownNature:
    drawdown = _get(row, "drawdown_52w")
    excess = _get(row, "excess_drawdown_vs_sector")
    if drawdown is None:
        return "unknown"
    if drawdown > t.oversold_drawdown:
        return "none"
    if excess is not None and excess <= t.idiosyncratic_excess:
        return "idiosyncratic"
    sector_drawdown = (
        _get(sector_row, "drawdown_52w") if sector_row is not None else None
    )
    if sector_drawdown is not None and sector_drawdown <= t.oversold_drawdown / 2:
        return "sector"
    if excess is not None and excess > t.idiosyncratic_excess:
        return "market"
    return "unknown"


def _evidence(row: pd.Series) -> list[Evidence]:  # noqa: C901, PLR0912
    items: list[Evidence] = []

    def add(
        signal: str, value: float | None, temporary: bool | None, detail: str
    ) -> None:
        reads = (
            "neutral"
            if temporary is None
            else ("temporary" if temporary else "structural")
        )
        items.append(
            Evidence(signal=signal, reads_as=reads, value=value, detail=detail)
        )

    surprise = _get(row, "eps_surprise_last")
    since = _get(row, "return_since_earnings")
    if surprise is not None and since is not None and since < -0.08:
        add(
            "earnings_reaction",
            surprise,
            surprise >= 0,
            (
                "Fell after a beat (overreaction)"
                if surprise >= 0
                else "Fell after a miss"
            ),
        )
    target = _get(row, "target_mean_change_3m")
    ret_3m = _get(row, "ret_3m")
    if target is not None and ret_3m is not None:
        add(
            "analyst_targets",
            target,
            target > ret_3m,
            (
                "Targets cut less than the price"
                if target > ret_3m
                else "Targets cut faster than the price"
            ),
        )
    buy_change = _get(row, "buy_ratio_change_3m")
    if buy_change is not None:
        add(
            "recommendations",
            buy_change,
            None if abs(buy_change) < 0.05 else buy_change > 0,
            "Buy ratio trend over three months",
        )
    insiders = _get(row, "insider_net_buy_3m")
    if insiders is not None and insiders != 0:
        add(
            "insiders",
            insiders,
            insiders > 0,
            (
                "Insiders bought after the drop"
                if insiders > 0
                else "Insiders sold into the drop"
            ),
        )
    cluster = _get(row, "insider_cluster_buy")
    if cluster:
        add(
            "insider_cluster",
            cluster,
            True,
            "At least three insiders bought within 30 days",
        )
    holders = _get(row, "inst_holders_change")
    if holders is not None and holders != 0:
        add(
            "institutions",
            holders,
            holders > 0,
            "Net new 13F holders" if holders > 0 else "Holders leaving",
        )
    activist = _get(row, "activist_13d")
    if activist:
        add("activist", activist, True, "A 13D holder appeared")
    short_change = _get(row, "short_interest_change_1m")
    if short_change is not None:
        add(
            "short_interest",
            short_change,
            None if abs(short_change) < 0.1 else short_change < 0,
            "Short interest trend over a month",
        )
    flags = _get(row, "red_flag_8k_12m")
    if flags:
        add(
            "8k_structural",
            flags,
            False,
            "Restatement, auditor change, delisting notice, bankruptcy or debt acceleration filed",
        )
    departures = _get(row, "officer_departure_12m")
    if departures:
        add("8k_departure", departures, False, "Officer or director departure filed")
    offering = _get(row, "equity_offering_12m")
    if offering:
        add("equity_offering", offering, False, "Raised equity in the last year")
    gap = _fundamental_price_gap(row)
    if gap is not None:
        add(
            "fundamental_price_gap",
            gap,
            gap > 0,
            (
                "Fundamentals rank above the price momentum"
                if gap > 0
                else "Price is in line with fundamentals"
            ),
        )
    pe_z = _get(row, "pe_vs_5y_z")
    margin = _get(row, "operating_margin_trend")
    if pe_z is not None and margin is not None:
        add(
            "valuation_vs_history",
            pe_z,
            pe_z < -0.5 and margin >= -0.02,
            "Cheap against history while margins hold",
        )
    return items


def _fundamental_price_gap(row: pd.Series) -> float | None:
    fundamentals = [
        v
        for v in (
            _get(row, "pct_roic"),
            _get(row, "pct_fcf_conversion"),
            _get(row, "pct_piotroski_f"),
        )
        if v is not None
    ]
    price = _get(row, "pct_mom_12_1")
    if not fundamentals or price is None:
        return None
    return round(float(np.mean(fundamentals)) - price, 1)


def _recovery(row: pd.Series) -> list[str]:
    signals = []
    if _get(row, "rsi_turning_up"):
        signals.append("RSI turning up from oversold")
    if _get(row, "above_20d_high"):
        signals.append("Price back above the 20-day high")
    ratio = _get(row, "up_down_volume_20d")
    if ratio is not None and ratio > 1.2:
        signals.append("Up-day volume exceeds down-day volume")
    if _get(row, "insider_cluster_buy"):
        signals.append("Insider cluster buy")
    surprise = _get(row, "eps_surprise_last")
    if (
        surprise is not None
        and surprise > 0
        and (_get(row, "days_since_earnings") or 999) < 45
    ):
        signals.append("Positive surprise at the last report")
    target = _get(row, "target_mean_change_1m")
    if target is not None and target > 0:
        signals.append("Targets rising over the last month")
    return signals


def _label(
    sound: bool | None, oversold: bool | None, balance: float | None, flags: list[str]
) -> Label:
    if sound is None:
        return "insufficient_data"
    if oversold and sound and (balance is None or balance >= 0) and not flags:
        return "quality_on_sale"
    if oversold and (flags or (balance is not None and balance < 0) or not sound):
        return "falling_knife" if (flags or not sound) else "oversold_unclear"
    if sound:
        return "sound_not_cheap"
    return "avoid"


def _likely_cause(  # noqa: PLR0911
    row: pd.Series, nature: DrawdownNature, evidence: list[Evidence]
) -> str:
    by_signal = {e.signal: e for e in evidence}
    if "8k_structural" in by_signal:
        return "structural event filed on 8-K"
    if "equity_offering" in by_signal:
        return "equity offering"
    if "earnings_reaction" in by_signal:
        return (
            "earnings miss"
            if by_signal["earnings_reaction"].reads_as == "structural"
            else "overreaction to earnings"
        )
    if nature == "sector":
        return "sector sell-off"
    if nature == "market":
        return "market sell-off"
    if nature == "idiosyncratic":
        return "company-specific, cause not identified in the data"
    return "no material drawdown" if nature == "none" else "unknown"


def assess(  # noqa: PLR0913
    symbol: str,
    row: pd.Series,
    as_of: str,
    sector_row: pd.Series | None = None,
    thresholds: VerdictThresholds | None = None,
    coverage_grade: str | None = None,
) -> StockVerdict:
    t = thresholds or VerdictThresholds()
    checks = fundamental_checks(row, t)
    signals = oversold_signals(row, t)
    fundamental_score = _score(checks)
    oversold_score = _score(signals)
    sound = None if fundamental_score is None else fundamental_score >= t.sound_score
    oversold = None if oversold_score is None else oversold_score >= t.oversold_score
    evidence = _evidence(row)
    temporary = [e for e in evidence if e.reads_as == "temporary"]
    structural = [e for e in evidence if e.reads_as == "structural"]
    directed = len(temporary) + len(structural)
    balance = (
        round(100 * (len(temporary) - len(structural)) / directed, 1)
        if directed
        else None
    )
    flags = [
        e.detail for e in structural if e.signal in {"8k_structural", "equity_offering"}
    ]
    nature = _nature(row, sector_row, t)
    label = _label(sound, oversold, balance, flags)
    cause = _likely_cause(row, nature, evidence)
    fundamentals = "n/a" if fundamental_score is None else fundamental_score
    oversold_text = "n/a" if oversold_score is None else oversold_score
    summary = (
        f"{symbol}: {label.replace('_', ' ')}. Fundamentals {fundamentals}/100, "
        f"oversold {oversold_text}/100, drawdown {nature}, cause: {cause}."
    )
    return StockVerdict(
        symbol=symbol,
        as_of=as_of,
        label=label,
        fundamentally_sound=sound,
        fundamental_score=fundamental_score,
        fundamental_checks=checks,
        oversold=oversold,
        oversold_score=oversold_score,
        oversold_signals=signals,
        drawdown_nature=nature,
        likely_cause=cause,
        temporary_vs_structural=balance,
        structural_red_flags=flags
        + [e.detail for e in structural if e.signal == "8k_departure"],
        recovery_evidence=_recovery(row),
        evidence=evidence,
        data_coverage_grade=coverage_grade,
        summary=summary,
    )
