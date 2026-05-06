from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskFamily(StrEnum):
    SINGLE_LOOKUP = "single_lookup"
    SMALL_PARALLEL_LOOKUP = "small_parallel_lookup"
    SCALAR_LARGE_FANOUT = "scalar_large_fanout"
    BATCH_LARGE_FANOUT = "batch_large_fanout"
    DEEP_BRANCHING_FILTER_RANK = "deep_branching_filter_rank"


class ToolShape(StrEnum):
    SCALAR = "scalar"
    BATCH = "batch"


class FailureCategory(StrEnum):
    PROVIDER_FAILURE = "provider_failure"
    MODEL_PROTOCOL_ERROR = "model_protocol_error"
    SCHEMA_FAILURE = "schema_failure"
    SCORING_FAILURE = "scoring_failure"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"
    ADAPTER_FAILURE = "adapter_failure"
    HARNESS_FAILURE = "harness_failure"


class ScoreFailureReason(StrEnum):
    SCHEMA_INVALID = "schema_invalid"
    TASK_ID_MISMATCH = "task_id_mismatch"
    TIMEOUT = "timeout"


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 1
    task_family: TaskFamily = TaskFamily.SCALAR_LARGE_FANOUT
    tool_shape: ToolShape = ToolShape.SCALAR
    shard_count: int = Field(default=5, ge=1)
    candidates_per_shard: int = Field(default=20, ge=1)
    payload_bytes: int = Field(default=256, ge=0)
    relevant_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1)

    @property
    def candidate_count(self) -> int:
        return self.shard_count * self.candidates_per_shard


class ProbeTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    workload: WorkloadConfig
    max_tool_calls: int = Field(default=200, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    shard_id: int = Field(ge=0)
    title: str
    category: str
    age_days: int = Field(ge=0)
    approvals: int = Field(ge=0)
    failing_checks: int = Field(ge=0)
    reactions: int = Field(ge=0)
    changed_files: int = Field(ge=0)
    is_draft: bool = False
    is_bot_authored: bool = False
    relevance: float = Field(ge=0.0, le=1.0)
    payload: str = ""


class RankedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    score: float
    rationale: str | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    candidates: list[RankedCandidate]


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_valid: bool
    timed_out: bool = False
    top_k_overlap: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    ndcg_at_k: float = Field(ge=0.0, le=1.0)
    failure_reason: ScoreFailureReason | None = None


class UsageStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    failed_tool_calls: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    tool_response_bytes_total: int = Field(default=0, ge=0)
    model_visible_bytes_total: int | None = Field(default=None, ge=0)


class TraceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    span_count: int = Field(default=0, ge=0)
    nested_tool_call_count: int = Field(default=0, ge=0)
    failure_category: FailureCategory | None = None


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    response_bytes: int = Field(ge=0)
    model_visible: bool
    item_count: int = Field(default=0, ge=0)


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    model_visible: bool = True


class NormalizedToolRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class NormalizedToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: NormalizedToolRequest
    result: Any | None = None
    error: str | None = None


class NormalizedModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)


class ModelTurnRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: ProbeTask
    turn_index: int = Field(ge=1)
    tool_results: list[NormalizedToolResult] = Field(default_factory=list)


class ModelTurnResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_requests: list[NormalizedToolRequest] = Field(default_factory=list)
    final_answer: StructuredAnswer | dict[str, Any] | None = None
    usage: NormalizedModelUsage = Field(default_factory=NormalizedModelUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: StructuredAnswer | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    trace: TraceSummary = Field(default_factory=TraceSummary)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ResultProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    task_hash: str | None = None
    prompt_hash: str | None = None
    tool_spec_hash: str | None = None
    candidate_set_hash: str | None = None
    oracle_answer_hash: str | None = None
    executor_name: str | None = None
    executor_config: dict[str, Any] = Field(default_factory=dict)
    benchmark_version: str | None = None


class ArmResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    arm_name: str
    repetition: int = Field(ge=1)
    trial_id: str | None = None
    arm_order_index: int | None = Field(default=None, ge=0)
    arm_order: tuple[str, ...] = Field(default_factory=tuple)
    latency_ms: float = Field(ge=0.0)
    timed_out: bool = False
    provenance: ResultProvenance = Field(default_factory=ResultProvenance)
    execution: ExecutionResult
    score: ScoreResult
