from datetime import datetime
from typing import Literal, get_args

import pandas as pd
from pydantic import BaseModel, Field

JobType = Literal["Update data", "Update analysis", "Fetching news"]
JobStatus = Literal["Completed", "Failed", "Running", "Started"]
StatusIcon = ["✅ Completed", "❌ Failed", "🔄 Running", "🚀 Started"]


class JobTrackerStatus(BaseModel):
    job_id: str
    status: JobStatus = "Started"


class JobTracker(JobTrackerStatus):
    type: JobType
    started_at: datetime = Field(default_factory=datetime.now)


def add_icons(data: pd.DataFrame) -> pd.DataFrame:
    status_map = dict(zip(list(get_args(JobStatus)), StatusIcon, strict=True))
    data["status"] = data["status"].map(status_map)
    return data
