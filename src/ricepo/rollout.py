"""Native-token rollout and counterfactual branches with explicit experiment rules.

No prompt template, query separator, stopping rule, reward aggregation, or failure
reward is supplied here. Those choices must come from the caller's verified spec.
Counterfactual trajectories are returned separately for auditing; they are not
automatically added to the policy optimization batch.

Credit-estimation branches require an original trajectory at the full five-step
horizon and continue to that same maximum depth (paper Section 4.2.2). Early
stopping remains available for other rollout uses, but is rejected for branching
so an original R_3 is never compared against counterfactual R_5 rewards.

Nonempty reasoning/summary spans and summary text are this implementation's
format policy, not a requirement established by the paper. Appendix A.2 allows
a zero-word summary; its stop/reward/entropy treatment is not specified here.
Dataset exclusion IDs are fixed per query. Previously retrieved documents are
allowed to appear again, including in counterfactual branches.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType

from .contracts import (
    Anchor,
    BranchGroup,
    BranchOutcome,
    Generation,
    Generator,
    QueryComposer,
    RankedDocument,
    RetrievalResult,
    RetrievalStep,
    Retriever,
    RewardScorer,
    Span,
    Trajectory,
    InvalidGeneration,
    FailureAction,
    generation_action_mask,
)


from .audit import operation, fingerprint, saved_item, logical_request_id

MAX_RETRIEVAL_DEPTH = 5


class RolloutError(RuntimeError):
    """An invalid rollout must be investigated, never assigned a made-up reward."""


class PolicyVersionError(RolloutError):
    """The generator is not the frozen old policy requested by the caller."""


class MalformedGenerationError(RolloutError):
    """Native token output or its supplied segmentation is invalid."""
    code = "tag_structure"


class RolloutLengthError(RolloutError):
    """A configured generation/context limit was exceeded."""


class RetrievalContractError(RolloutError):
    """A retrieval response violated the exclusion or native-token contract."""


@dataclass(frozen=True)
class RolloutRules:
    accepted_finish_reasons: frozenset[str]
    max_new_tokens: int
    max_context_tokens: int
    depth: int = MAX_RETRIEVAL_DEPTH

    def __post_init__(self) -> None:
        if not self.accepted_finish_reasons or any(
            not isinstance(reason, str) or not reason
            for reason in self.accepted_finish_reasons
        ):
            raise ValueError("explicit accepted_finish_reasons are required")
        for name in ("max_new_tokens", "max_context_tokens", "depth"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be an explicit positive integer")


@dataclass(frozen=True)
class BranchRequest:
    trajectory_id: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class BranchRollouts:
    group: BranchGroup
    # Statistics/audit data only. The original training batch remains separate.
    trajectories: tuple[Trajectory, ...]


def _tokens(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    try:
        ids = tuple(values)
    except TypeError as exc:
        raise MalformedGenerationError(f"{label} must be a token ID sequence") from exc
    if any(type(value) is not int or value < 0 for value in ids):
        raise MalformedGenerationError(f"{label} must contain nonnegative integer token IDs")
    return ids


def _finite_reward(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise RolloutError(f"{label} must be a finite numeric reward")
    if not 0 <= value <= 1:
        raise RolloutError(f"{label} must use fractional NDCG scale [0, 1]")
    return float(value)


class RolloutRunner:
    def __init__(
        self,
        *,
        generator: Generator,
        retriever: Retriever,
        reward_scorer: RewardScorer,
        query_composer: QueryComposer,
        rules: RolloutRules,
        should_stop: Callable[[tuple[RetrievalStep, ...]], bool],
        final_reward: Callable[[tuple[RetrievalStep, ...]], float],
        excluded_ids_by_query: Mapping[str, frozenset[str]] | None = None,
        render_next_turn: Callable | None = None,
        failure_semantics: str = "raise",
    ) -> None:
        self.generator = generator
        if failure_semantics not in ("raise","invalid_action_terminal_zero_v1"):
            raise ValueError("unsupported failure semantics")
        self.failure_semantics = failure_semantics
        self.render_next_turn = render_next_turn
        self.retriever = retriever
        self.reward_scorer = reward_scorer
        self.query_composer = query_composer
        self.rules = rules
        self.should_stop = should_stop
        self.final_reward = final_reward
        exclusions = {}
        for query_id, document_ids in (excluded_ids_by_query or {}).items():
            if not isinstance(query_id, str) or not query_id:
                raise ValueError("exclusion query IDs must be nonempty strings")
            if not isinstance(document_ids, frozenset) or any(
                not isinstance(document_id, str) or not document_id for document_id in document_ids
            ):
                raise ValueError("each dataset exclusion set must be a frozenset of document IDs")
            exclusions[query_id] = document_ids
        self.excluded_ids_by_query = MappingProxyType(exclusions)

    def _check_policy(self, expected: str) -> None:
        if not expected or self.generator.policy_version != expected:
            raise PolicyVersionError(f"generator policy must remain frozen at {expected!r}")

    def _check_length(self, ids: Sequence[int]) -> None:
        if len(ids) > self.rules.max_context_tokens:
            raise RolloutLengthError("native-token history exceeds max_context_tokens")

    def _validate_generation(self, generation: Generation) -> tuple[int, ...]:
        if not isinstance(generation, Generation):
            raise MalformedGenerationError("generator must return Generation")
        ids = _tokens(generation.token_ids, label="generation")
        if not ids:
            raise MalformedGenerationError("empty generation")
        try:
            action_count=sum(generation_action_mask(generation))
        except ValueError as exc:
            raise MalformedGenerationError(str(exc)) from exc
        if action_count > self.rules.max_new_tokens:
            raise RolloutLengthError("generation exceeds max_new_tokens")
        if generation.finish_reason not in self.rules.accepted_finish_reasons:
            raise MalformedGenerationError(
                f"unaccepted finish reason: {generation.finish_reason!r}"
            )
        reasoning, summary = generation.reasoning_span, generation.summary_span
        if not isinstance(reasoning, Span) or not isinstance(summary, Span):
            raise MalformedGenerationError("parser must supply native-token Span objects")
        if any(type(offset) is not int for offset in (
            reasoning.start, reasoning.end, summary.start, summary.end
        )):
            raise MalformedGenerationError("span offsets must be integer native-token positions")
        if not (0 <= reasoning.start <= reasoning.end <= summary.start < summary.end <= len(ids)):
            raise MalformedGenerationError(
                "format policy requires nonempty, ordered, in-bounds reasoning and summary spans"
            )
        if not isinstance(generation.summary_text, str) or not generation.summary_text:
            raise MalformedGenerationError("format policy requires nonempty summary_text")
        return ids

    @staticmethod
    def _validate_seeds(seeds: Sequence[int], count: int) -> tuple[int, ...]:
        values = tuple(seeds)
        if len(values) != count or any(type(seed) is not int or seed < 0 for seed in values):
            raise ValueError(f"supply exactly {count} explicit nonnegative integer seeds")
        return values

    def _validate_saved_history(self, original: Trajectory) -> None:
        if self.failure_semantics == "invalid_action_terminal_zero_v1":
            from .failure_zero import validate_trajectory
            validate_trajectory(original,self.rules.depth)
            return
        history = _tokens(original.steps[0].prefix_token_ids, label="saved initial prefix")
        if not history:
            raise ValueError("original trajectory must have a nonempty initial prefix")
        mask = (False,) * len(history)
        for step in original.steps:
            if step.prefix_token_ids != history:
                raise ValueError("original step prefix does not match its complete saved history")
            generated = self._validate_generation(step.generation)
            offset = len(history)
            relative_reasoning = step.generation.reasoning_span
            relative_summary = step.generation.summary_span
            if (step.reasoning_span != Span(offset + relative_reasoning.start, offset + relative_reasoning.end)
                    or step.summary_span != Span(offset + relative_summary.start, offset + relative_summary.end)):
                raise ValueError("original absolute spans do not match native generation offsets")
            observation = _tokens(step.retrieval.observation_token_ids, label="saved observation")
            history += generated + observation
            mask += generation_action_mask(step.generation) + (False,) * len(observation)
        if history != original.token_ids or mask != original.generated_mask:
            raise ValueError("original trajectory IDs or mask do not match its saved steps")
        if any(type(value) is not bool for value in original.generated_mask):
            raise ValueError("original generated mask must contain boolean values")
        self._check_length(history)

    def _continue(
        self,
        *,
        query_id: str,
        trajectory_id: str,
        original_query: str,
        policy_version: str,
        token_ids: tuple[int, ...],
        generated_mask: tuple[bool, ...],
        previous_steps: tuple[RetrievalStep, ...],
        seeds: tuple[int, ...],
        final_depth: int,
        honor_stop: bool,
        initial_ndcg: float | None = None,
    ) -> Trajectory:
        self._check_policy(policy_version)
        history = token_ids
        mask = generated_mask
        steps = list(previous_steps)
        excluded = self.excluded_ids_by_query.get(query_id, frozenset())
        termination = "depth_limit" if honor_stop else "counterfactual_max_depth"
        for depth, seed in zip(range(len(steps) + 1, final_depth + 1), seeds, strict=True):
            prefix = history
            self._check_length(prefix)
            self._check_policy(policy_version)
            kind = "original_generation" if honor_stop else "branch_generation"
            request_id = logical_request_id(policy_version, query_id, trajectory_id, depth, seed)
            replay = getattr(self.generator,"action_replay",None) or getattr(self.generator,"completion_replay",None)
            saved_native_replay = bool(replay and replay(prefix,seed,policy_version) is not None)
            with operation(kind, request_id=request_id, query_id=query_id, trajectory_id=trajectory_id,
                           seed=seed, depth=depth, input_tokens=len(prefix), policy_version=policy_version,
                           saved_native_replay=saved_native_replay,prefix_sha256=fingerprint(prefix)):
                generation = self.generator.generate(prefix, seed=seed)
            self._check_policy(policy_version)
            if isinstance(generation,InvalidGeneration):
                if self.failure_semantics != "invalid_action_terminal_zero_v1":
                    raise MalformedGenerationError("unexpected invalid-action result")
                ids = _tokens(generation.token_ids,label="failed native generation")
                if not ids or sum(generation_action_mask(generation))>self.rules.max_new_tokens:
                    raise MalformedGenerationError("invalid failure evidence size")
                self._check_length(prefix+ids)
                last = steps[-1].local_reward if steps else initial_ndcg
                _finite_reward(last,label="last valid retrieval NDCG")
                failure = FailureAction(depth,prefix,generation,seed,policy_version)
                return Trajectory(query_id,trajectory_id,policy_version,prefix+ids,
                    mask+generation_action_mask(generation),tuple(steps),0.0,"invalid_action_terminal_zero_v1",
                    failure,initial_ndcg,last,0.0)
            generated_ids = self._validate_generation(generation)
            self._check_length(prefix + generated_ids)
            query = self.query_composer.compose(original_query, generation.summary_text)
            if not isinstance(query, str) or not query:
                raise RetrievalContractError("query composer must return a nonempty string")
            with operation("retrieval_followup", query_id=query_id, trajectory_id=trajectory_id,
                           depth=depth, request_id=request_id, query=query, excluded_ids=sorted(excluded)):
                retrieval = self.retriever.retrieve(query, excluded_ids=frozenset(excluded))
            if not isinstance(retrieval, RetrievalResult):
                raise RetrievalContractError("retriever must return RetrievalResult")
            if retrieval.query != query:
                raise RetrievalContractError("retriever changed the composed query")
            if any(not isinstance(doc, RankedDocument) for doc in retrieval.documents):
                raise RetrievalContractError("retriever must return RankedDocument records")
            ranked_ids = tuple(doc.document_id for doc in retrieval.documents)
            if any(not isinstance(doc_id, str) or not doc_id for doc_id in ranked_ids):
                raise RetrievalContractError("retrieval document IDs must be nonempty strings")
            if len(set(ranked_ids)) != len(ranked_ids):
                raise RetrievalContractError("duplicate document IDs in retrieval response")
            if excluded.intersection(ranked_ids):
                raise RetrievalContractError("retriever returned an excluded document ID")
            if any(isinstance(doc.score, bool) or not isinstance(doc.score, (int, float))
                   or not isfinite(doc.score) for doc in retrieval.documents):
                raise RetrievalContractError("retriever returned a nonfinite document score")
            if self.render_next_turn is not None:
                retrieval = replace(retrieval, observation_token_ids=self.render_next_turn(
                    retrieval.documents, generated_ids))
            observation = _tokens(retrieval.observation_token_ids, label="retrieval observation")
            local_reward = _finite_reward(
                self.reward_scorer.score(query_id, ranked_ids), label="local reward"
            )
            step = RetrievalStep(
                depth=depth,
                prefix_token_ids=prefix,
                generation=generation,
                retrieval=retrieval,
                local_reward=local_reward,
                reasoning_span=Span(len(prefix) + generation.reasoning_span.start,
                                    len(prefix) + generation.reasoning_span.end),
                summary_span=Span(len(prefix) + generation.summary_span.start,
                                  len(prefix) + generation.summary_span.end),
            )
            steps.append(step)
            history = prefix + generated_ids + observation
            mask += generation_action_mask(generation) + (False,) * len(observation)
            self._check_length(history)
            if honor_stop:
                stop = self.should_stop(tuple(steps))
                if type(stop) is not bool:
                    raise RolloutError("should_stop must return an explicit boolean")
                if stop:
                    termination = "caller_stop"
                    break
        self._check_policy(policy_version)
        frozen_steps = tuple(steps)
        reward = _finite_reward(self.final_reward(frozen_steps), label="final reward")
        self._check_policy(policy_version)
        return Trajectory(
            query_id=query_id,
            trajectory_id=trajectory_id,
            policy_version=policy_version,
            token_ids=history,
            generated_mask=mask,
            steps=frozen_steps,
            final_reward=reward,
            termination=termination,
            initial_ndcg=initial_ndcg,
            last_valid_ndcg=reward if self.failure_semantics=="invalid_action_terminal_zero_v1" else None,
        )

    def run(
        self,
        *,
        query_id: str,
        trajectory_id: str,
        original_query: str,
        initial_prefix_token_ids: Sequence[int],
        seeds: Sequence[int],
        old_policy_version: str,
        initial_ndcg: float | None = None,
    ) -> Trajectory:
        """Run at most five retrievals; the caller explicitly defines early stopping."""
        if not query_id or not trajectory_id or not original_query:
            raise ValueError("query_id, trajectory_id, and original_query are required")
        prefix = _tokens(initial_prefix_token_ids, label="initial prefix")
        if not prefix:
            raise MalformedGenerationError("a nonempty native-token initial prefix is required")
        return self._continue(
            query_id=query_id,
            trajectory_id=trajectory_id,
            original_query=original_query,
            policy_version=old_policy_version,
            token_ids=prefix,
            generated_mask=(False,) * len(prefix),
            previous_steps=(),
            seeds=self._validate_seeds(seeds, self.rules.depth),
            final_depth=self.rules.depth,
            honor_stop=True,
            initial_ndcg=initial_ndcg,
        )

    def branch(
        self,
        *,
        original_query: str,
        original: Trajectory,
        anchor: Anchor,
        requests: Sequence[BranchRequest],
        batched: bool = False,
    ) -> BranchRollouts:
        """Regenerate from the exact saved prefix, retaining the observed outcome.

        The original must have reached the maximum depth of five; every new
        branch also reaches five. An early-stopped original is rejected rather
        than silently changing its recorded final reward. Seeds and
        branch IDs are explicit and distinct; a stateful generator must honor each
        supplied seed independently. New trajectories are statistics/audit data.
        """
        if (anchor.query_id, anchor.trajectory_id) != (original.query_id, original.trajectory_id):
            raise ValueError("anchor does not identify the original trajectory")
        final_depth = self.rules.depth if self.failure_semantics=="invalid_action_terminal_zero_v1" else len(original.steps)
        if not 1 <= anchor.depth <= len(original.steps) <= final_depth <= self.rules.depth:
            raise ValueError("anchor/original depth is outside the configured rollout")
        if final_depth != self.rules.depth:
            raise ValueError(f"credit branches require an original at the full {self.rules.depth}-step horizon")
        if tuple(step.depth for step in original.steps) != tuple(range(1, len(original.steps) + 1)):
            raise ValueError("original steps must have contiguous one-based depths")
        self._validate_saved_history(original)
        selected = original.steps[anchor.depth - 1]
        if (anchor.reasoning_span, anchor.summary_span) != (selected.reasoning_span, selected.summary_span):
            raise ValueError("anchor spans do not match the original saved step")
        prefix = _tokens(selected.prefix_token_ids, label="saved prefix")
        if not prefix or original.token_ids[:len(prefix)] != prefix:
            raise ValueError("saved pre-reasoning prefix does not match the original trajectory")
        if len(original.generated_mask) != len(original.token_ids):
            raise ValueError("original token IDs and generated mask differ in length")
        if not original_query:
            raise ValueError("original_query is required")
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one new branch must be explicitly requested")
        branch_ids = tuple(request.trajectory_id for request in requests)
        if (any(not branch_id for branch_id in branch_ids)
                or len(set(branch_ids)) != len(branch_ids)
                or original.trajectory_id in branch_ids):
            raise ValueError("new branch IDs must be nonempty and distinct from the original")
        seed_sets = tuple(
            self._validate_seeds(request.seeds, final_depth - anchor.depth + 1)
            for request in requests
        )
        all_seeds = tuple(seed for seeds in seed_sets for seed in seeds)
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("new branches require distinct per-step seeds")
        self._check_policy(original.policy_version)
        outcomes = [BranchOutcome(
            branch_id=original.trajectory_id,
            is_original=True,
            local_reward=_finite_reward(selected.local_reward, label="original local reward"),
            final_reward=_finite_reward(original.final_reward, label="original final reward"),
            policy_version=original.policy_version,
            prefix_token_ids=prefix,
        )]
        trajectories = []
        batch_results = None
        if batched:
            from .batched_rollout import continue_many
            states=[dict(query_id=original.query_id,trajectory_id=request.trajectory_id,
                original_query=original_query,policy_version=original.policy_version,token_ids=prefix,
                generated_mask=original.generated_mask[:len(prefix)],previous_steps=original.steps[:anchor.depth-1],
                seeds=seeds,final_depth=final_depth,honor_stop=False,initial_ndcg=original.initial_ndcg)
                for request,seeds in zip(requests,seed_sets,strict=True)]
            batch_results=iter(continue_many(self,states))
        for request, seeds in zip(requests, seed_sets, strict=True):
            trajectory = next(batch_results) if batch_results is not None else self._continue(
                query_id=original.query_id,
                trajectory_id=request.trajectory_id,
                original_query=original_query,
                policy_version=original.policy_version,
                token_ids=prefix,
                generated_mask=original.generated_mask[:len(prefix)],
                previous_steps=original.steps[:anchor.depth - 1],
                seeds=seeds,
                final_depth=final_depth,
                honor_stop=False,
                initial_ndcg=original.initial_ndcg,
            )
            saved_item('counterfactual', trajectory.trajectory_id, trajectory)
            trajectories.append(trajectory)
            outcomes.append(BranchOutcome(
                branch_id=request.trajectory_id,
                is_original=False,
                local_reward=(trajectory.steps[anchor.depth - 1].local_reward
                              if len(trajectory.steps)>=anchor.depth else None),
                final_reward=trajectory.final_reward,
                policy_version=trajectory.policy_version,
                prefix_token_ids=prefix,
            ))
        return BranchRollouts(BranchGroup(anchor=anchor, outcomes=tuple(outcomes)), tuple(trajectories))
