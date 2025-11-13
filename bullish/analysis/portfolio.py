from typing import List, Optional, Any

from pydantic import BaseModel, Field


class PortfolioNewAsset(BaseModel):
    symbol: str


class PortfolioAsset(PortfolioNewAsset):
    value: float


class Portfolio(BaseModel):
    name: str = "default"
    current_assets: List[PortfolioAsset] = Field(
        default_factory=lambda: [PortfolioAsset(symbol="", value=1000)]
    )
    new_assets: List[PortfolioNewAsset] = Field(default_factory=list)
    amount: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude={"name"})

    def default_name(self) -> bool:
        return self.name == "default"

    def valid(self) -> bool:
        return self.current_assets is not None or self.new_assets is not None
