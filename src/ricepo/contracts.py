"""Shared immutable records. All spans are half-open native token offsets."""

from dataclasses import dataclass
import math
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
class GenerationStage:
    name: str
    token_span: Span
    finish_reason: str
    rng_before_sha256: str | None = None
    rng_after_sha256: str | None = None


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    reasoning_span: Span
    summary_span: Span
    summary_text: str
    finish_reason: str
    action_mask: tuple[bool, ...] | None = None
    protocol: str = "legacy_single_stage_v1"
    sampled_log_probs: tuple[float | None, ...] | None = None
    stages: tuple[GenerationStage, ...] = ()


@dataclass(frozen=True)
class InvalidGeneration:
    token_ids: tuple[int, ...]
    finish_reason: str
    category: str
    reason: str
    action_mask: tuple[bool, ...] | None = None
    protocol: str = "legacy_single_stage_v1"
    sampled_log_probs: tuple[float | None, ...] | None = None
    stages: tuple[GenerationStage, ...] = ()


def generation_from_record(record):
    """Deserialize native evidence without tokenization or text repair."""
    fields=dict(record)
    fields['token_ids']=tuple(fields['token_ids'])
    for key in ('action_mask','sampled_log_probs'):
        if fields.get(key) is not None:fields[key]=tuple(fields[key])
    fields['stages']=tuple(GenerationStage(**{**s,'token_span':Span(**s['token_span'])})
                           for s in fields.get('stages',()))
    cls=InvalidGeneration if 'category' in fields else Generation
    if cls is Generation:
        for key in ('reasoning_span','summary_span'):fields[key]=Span(**fields[key])
    result=cls(**fields)
    generation_action_mask(result)
    return result


def generation_action_mask(generation):
    """Legacy means sampled IDs only. Mixed histories require an explicit protocol."""
    size=len(generation.token_ids)
    mask=generation.action_mask
    if generation.protocol == "legacy_single_stage_v1":
        if generation.stages or mask is not None and mask != (True,)*size:
            raise ValueError("legacy generation cannot hide environment tokens")
        mask=(True,)*size
    elif generation.protocol == "two_stage_env_summary_v1":
        if mask is None or generation.sampled_log_probs is None or not generation.stages:
            raise ValueError("two-stage generation needs explicit mask, probabilities and stages")
        stages=generation.stages
        if len(stages)>2 or stages[0].name!='think' or stages[0].token_span.start!=0:
            raise ValueError("invalid two-stage identity")
        if len(stages)==2 and (stages[1].name!='summary' or stages[1].token_span.start<=stages[0].token_span.end):
            raise ValueError("Summary requires a separate environment control interval")
        expected=[False]*size
        for stage in stages:
            a,b=stage.token_span.start,stage.token_span.end
            if not 0<=a<b<=size:raise ValueError("invalid stage native span")
            expected[a:b]=[True]*(b-a)
        if stages[-1].token_span.end!=size or tuple(expected)!=mask:
            raise ValueError("stage native intervals disagree with action mask")
    else:
        raise ValueError("unknown generation protocol")
    if len(mask)!=size or any(type(m) is not bool for m in mask):
        raise ValueError("action mask must align with every native token")
    probs=generation.sampled_log_probs
    if probs is not None:
        if len(probs)!=size:raise ValueError("behavior probability length differs")
        for action,lp in zip(mask,probs,strict=True):
            if action and (type(lp) not in (float,int) or not math.isfinite(lp) or lp>1e-5):
                raise ValueError("sampled action needs a finite conditional log probability")
            if not action and lp is not None:raise ValueError("environment token has no behavior action probability")
    if isinstance(generation,Generation):
        for span in (generation.reasoning_span,generation.summary_span):
            if not 0<=span.start<=span.end<=size or not all(mask[span.start:span.end]):
                raise ValueError("body span cannot contain environment control tokens")
    return mask


def trajectory_behavior_log_probs(trajectory):
    """Reconstruct aligned scalar evidence, never reconstruct native token IDs."""
    records=[(len(s.prefix_token_ids),s.generation) for s in trajectory.steps]
    if trajectory.failure:records.append((len(trajectory.failure.prefix_token_ids),trajectory.failure.generation))
    protocols={g.protocol for _,g in records}
    if len(protocols)>1:raise ValueError("mixed generation protocols within trajectory")
    if not records or protocols=={'legacy_single_stage_v1'}:return None
    result=[None]*len(trajectory.token_ids)
    for start,g in records:
        generation_action_mask(g)
        if trajectory.token_ids[start:start+len(g.token_ids)]!=g.token_ids:raise ValueError("behavior evidence native history differs")
        result[start:start+len(g.token_ids)]=g.sampled_log_probs
    if any(m!=(p is not None) for m,p in zip(trajectory.generated_mask,result,strict=True)):
        raise ValueError("trajectory behavior probabilities disagree with action mask")
    return tuple(0.0 if p is None else p for p in result)


@dataclass(frozen=True)
class FailureAction:
    depth: int
    prefix_token_ids: tuple[int, ...]
    generation: InvalidGeneration
    seed: int
    policy_version: str


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
    failure: FailureAction | None = None
    initial_ndcg: float | None = None
    last_valid_ndcg: float | None = None
    failure_terminal_reward: float | None = None


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
    local_reward: float | None
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

    def generate(self, prefix_token_ids: Sequence[int], *, seed: int) -> Generation | InvalidGeneration: ...


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
