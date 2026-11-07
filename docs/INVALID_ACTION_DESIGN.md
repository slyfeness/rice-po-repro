# invalid_action_terminal_v1 — proposal only

Historical proposal: the user rejected inherited-last-ranking training reward.
It is not enabled. The authorized independent environment is
[`invalid_action_terminal_zero_v1`](INVALID_ACTION_ZERO.md), which preserves
the last measured NDCG separately and assigns zero failure training utility.

Status: independently proposed failure semantics, unconfirmed by the RICE-PO
authors. Simulation only. Production training keeps its existing failure gate.

## Terminal transition

An invalid generated action contains missing, duplicate or nested tags, or
exhausts the generation ceiling before producing a valid action. Save its full
native IDs, sampled EOS if any, generation mask, prefix, seed, finish reason,
decoded text and structural reasons. Do not synthesize a Summary or tool call.
The generated IDs remain policy actions for trajectory-level PPO scoring.
Environment/control IDs remain context with a false direct-loss mask.

Maintain an observed retrieval state containing the actual last ranked IDs and
their NDCG@10, starting with D0. A valid Summary triggers the ordinary frozen
retriever and replaces this state. An invalid action ends further model/tool
requests and enters an absorbing failure state. Its ranking stays equal to the
last observed ranking. Final reward is the NDCG of that actual ranking. For a
first-action failure this is D0. No failure penalty or fabricated zero reward
is added. The environment's logical endpoint remains depth five, with zero
new actions or retrievals in the absorbing interval.

This choice means a failure can retain a high D0 score. It may encourage
preserving D0 rather than producing useful summaries. That incentive must be
reported and considered before any training approval. Stopping on invalid
actions is an explicit failure transition, not task natural completion.

## Missing Summary and local credit

Invalid actions have no Summary text, Summary body span, or entropy anchor.
No entropy value is imputed. Original sampling retains all scheduled
trajectories and their terminal states. For this proposed variant only, select
up to four anchors among actual valid summaries in each query group. If fewer
are available, record the reduced eligible budget. With zero valid anchors,
the query receives trajectory-relative advantages alone. This changes anchor
applicability from the strict reference and must be labeled independently.

For each existing valid anchor retain the original plus all five scheduled
counterfactual branches. A counterfactual that fails at the anchor receives
the environment-state reward inherited at that depth, labeled
inherited_actual_ranking. A later failure retains its most recent actual
ranking. Record whether each reward came from a new retrieval or inherited
state. Every scheduled outcome still participates once in population local
and residual variances at the common absorbing endpoint. Residual remains
final minus local. The .05/.03 gates, advantage replacement, and PPO expression
stay as written, but their failure-state inputs have this new definition.

No local advantage is assigned to a nonexistent Summary. Failed-action tokens
retain trajectory advantages. A valid preceding anchor may still receive its
local replacement and reasoning propagation according to the existing gate.
If semantic validity is ambiguous, reject the run rather than inventing a
parse. Infrastructure failures remain errors and receive no environment score.

## Controlled comparison

Both methods start from the same initial weights and use identical datasets,
query schedule, original trajectory count, generation settings, prompt,
message protocol, failure transitions, reward state and update budget.
Neither method discards failed originals or retries them. RICE-PO's five new
branches are additional credit-estimation cost. Report both scheduled and
executed actions, failures, retained D0 reward, actual retrieval counts,
absorbing depths, eligible anchors, optimizer steps and GPU hours.

Full-development coverage accounts for every query, including failure states,
under this proposed environment metric. Report valid-five-round completion
rate alongside it. Keep the strict Figure 5 evaluation and this environment
variant as separately named metrics. A retained-ranking failure reward cannot
be presented as a successfully generated query's retrieval result.

## Simulation evidence and limits

scripts/simulate_invalid_action.py exercises only the transition accounting.
Its small deterministic fixture verifies retention of native IDs, absence of
retrieval for invalid actions, inheritance of observed reward, and absorbing
termination. Fixture scores are labeled synthetic and never enter real runs.
No real-policy training, gradient, tokenizer or benchmark claim follows from
this simulation. Integration into production credit/trajectory types and
training remains subject to explicit approval of the above semantics.
