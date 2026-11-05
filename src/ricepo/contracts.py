"""Shared immutable records. All spans are half-open native token offsets."""

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid half-open token span")


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    score: float
    text: str = ""
    dense_raw: float | None = None
    sparse_raw: float | None = None
    dense_normalized: float | None = None
    sparse_normalized: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    documents: tuple[RankedDocument, ...]
    observation_token_ids: tuple[int, ...]
    backend_version: str


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    reasoning_span: Span
    summary_span: Span
    summary_text: str
    finish_reason: str


@dataclass(frozen=True)
class RetrievalStep:
    depth: int
    prefix_token_ids: tuple[int, ...]
    generation: Generation
    retrieval: RetrievalResult
    local_reward: float
    reasoning_span: Span
    summary_span: Span


@dataclass(frozen=True)
class Trajectory:
    query_id: str
    trajectory_id: str
    policy_version: str
    token_ids: tuple[int, ...]
    generated_mask: tuple[bool, ...]
    steps: tuple[RetrievalStep, ...]
    final_reward: float
    termination: str


@dataclass(frozen=True)
class Anchor:
    query_id: str
    trajectory_id: str
    depth: int
    entropy: float
    reasoning_span: Span
    summary_span: Span


@dataclass(frozen=True)
class BranchOutcome:
    branch_id: str
    is_original: bool
    local_reward: float
    final_reward: float
    policy_version: str
    prefix_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class BranchGroup:
    anchor: Anchor
    outcomes: tuple[BranchOutcome, ...]


@dataclass(frozen=True)
class CreditRecord:
    anchor: Anchor
    local_mean: float
    influence_variance: float
    residual_mean: float
    residual_variance: float
    propagate: bool
    summary_advantage: float


@dataclass(frozen=True)
class AdvantageBatch:
    trajectory_ids: tuple[str, ...]
    advantages: tuple[tuple[float, ...], ...]
    generated_masks: tuple[tuple[bool, ...], ...]
    records: tuple[CreditRecord, ...]


class Generator(Protocol):
    policy_version: str

    def generate(self, prefix_token_ids: Sequence[int], *, seed: int) -> Generation: ...


class Retriever(Protocol):
    def retrieve(self, query: str, *, excluded_ids: frozenset[str]) -> RetrievalResult: ...


class RewardScorer(Protocol):
    def score(self, query_id: str, ranked_ids: Sequence[str]) -> float: ...


class QueryComposer(Protocol):
    def compose(self, original_query: str, summary: str) -> str: ...


class EntropyProvider(Protocol):
    policy_version: str

    def token_entropies(self, trajectory: Trajectory) -> Sequence[float]: ...


LocalNormalization = Literal["center_only", "center_std"]
