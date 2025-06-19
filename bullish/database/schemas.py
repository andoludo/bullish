from sqlmodel import Field, SQLModel

from bullish.analysis import Analysis


class BaseTable(SQLModel):
    symbol: str = Field(primary_key=True)
    source: str = Field(primary_key=True)


class AnalysisORM(BaseTable, Analysis, table=True):
    __tablename__ = "analysis"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
