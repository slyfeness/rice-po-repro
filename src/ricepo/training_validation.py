"""Bounded paired GEMM validation; no paper-method or default runtime changes."""
from .artifacts import file_sha256, require_cluster_storage
from .state import read_json

KIND = 'tf32_gemm_training_validation_v1'


def validate_contract(candidate, frozen, recipe):
    expected = dict(method='ricepo',queries_per_update=4,total_updates=1 if candidate=='G_A' else 3,
        trajectories_per_query=4,anchors_per_query=4,total_branches_including_original=6,rollout_depth=5,
        training_seeds=[frozen['training_seed']],local_advantage_normalization='center_only',ppo_epochs=1,
        clip_epsilon=.2,actor_learning_rate=1e-6,actor_kl_coefficient=.001,influence_threshold=.05,
        residual_threshold=.03,checkpoint_every_updates=2,development_every_updates=4,checkpoint_selection='last',
        query_composition={'mode':'original_and_summary','separator':'\n'},max_new_tokens_per_step=4096,
        max_context_tokens=32768,temperature=1.,top_p=1.,top_k=0,loss_reduction='sequence_mean')
    if (candidate not in ('G_A','G_B') or frozen['kind']!=KIND
            or len(frozen['query_batches'])!=3 or any(len(b)!=4 or len(set(b))!=4 for b in frozen['query_batches'])
            or any(recipe.get(k)!=v for k,v in expected.items())):
        raise ValueError('configuration exceeds the frozen paired four-query validation')


def settings(spec, recipe):
    path=require_cluster_storage(spec['validation_plan_path'])
    if file_sha256(path)!=spec['validation_plan_sha256']:raise ValueError('validation plan changed')
    frozen=read_json(path);validate_contract(spec['candidate'],frozen,recipe)
    prior=read_json(require_cluster_storage(spec['qualification_result_path']))
    if (not prior['risk_screen_pass'] or prior['candidate']!='B'
            or prior['private_plan_sha256']!=frozen['old_qualification_plan_sha256']):
        raise ValueError('previous GEMM qualification does not match the frozen plan')
    return {**frozen,**spec,'query_batches':frozen['query_batches'][:recipe['total_updates']],
            'numeric_mode':'tf32_gemm_control' if spec['candidate']=='G_A' else 'tf32_gemm_pilot'}


def update_risk_limit(index):
    if type(index) is not int or not 0<=index<3:raise ValueError('bounded validation update index required')
    return .01 if index==0 else .02


def validate_committed_progress(record, *, index, policy, checkpoint, risk):
    expected=dict(completed_updates=index+1,policy_version=policy,
        checkpoint_complete_sha256=checkpoint,post_update_risk_sha256=risk)
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('committed validation position/weights/checkpoint/risk differs; never replay optimizer')
