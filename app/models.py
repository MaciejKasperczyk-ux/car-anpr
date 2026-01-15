from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class RunCreateRequest(BaseModel):
    cmd: str = Field(..., description="Command to run inside container workdir")
    env: Optional[Dict[str, str]] = Field(default=None, description="Extra env variables")
    name: Optional[str] = Field(default=None, description="Optional label for the run")


class RunCreateResponse(BaseModel):
    run_id: str


class RunInfo(BaseModel):
    run_id: str
    state: str
    exit_code: Optional[int] = None
    started_at: float
    finished_at: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class RunsList(BaseModel):
    runs: List[RunInfo]
