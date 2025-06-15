from sqlmodel import Field, SQLModel

from bullish.analysis import Analysis
from bullish.view import View


class BaseTable(SQLModel):
    symbol: str = Field(primary_key=True)
    source: str = Field(primary_key=True)


class AnalysisORM(BaseTable, Analysis, table=True):
    __tablename__ = "analysis"


class ViewORM(BaseTable, View, table=True):
    __tablename__ = "view"
