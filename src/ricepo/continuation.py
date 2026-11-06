"""Explicit checkpoint forks: immutable parents, local receipts, global sampling IDs."""
from pathlib import Path
import random

from .audit import fingerprint, stable_seed
from .artifacts import file_sha256
from .state import read_json, verify_checkpoint, selected_so_far


def continuation_recipe(parent, child):
    changed = sorted(k for k in parent.keys() | child.keys()
                     if k != 'total_updates' and parent.get(k) != child.get(k))
    if changed:
        raise ValueError('continuation recipe changed: '+', '.join(changed))
    if type(child.get('total_updates')) is not int or child['total_updates'] < 1:
        raise ValueError('child budget must be a positive number of additional updates')


def scheduled_queries(identities, seed, global_index, count):
    return random.Random(stable_seed(seed,'query_sampling',global_index)).sample(sorted(identities),count)


def validate_state_position(state, *, binding, iteration, seed, policy_version):
    expected = dict(binding=binding,iteration=iteration,seed=seed,policy_version=policy_version)
    if any(state.get(k) != v for k,v in expected.items()):
        raise ValueError('checkpoint optimizer identity disagrees with run position')


def parent_run(deployment, binding, plan, train):
    """Called inside Slurm. Full parent asset hashes are verified, never rewritten."""
    if not deployment.get('parent_run_root'):
        return None
    root = Path(deployment['parent_run_root']).resolve()
    if root == Path(deployment['output_root']).resolve():
        raise ValueError('child must have an independent output directory')
    parent = read_json(root/'binding.json')
    bound = fingerprint(parent)
    finished = read_json(root/'completed.json')
    # Deliberately bounded first implementation: fork a completed one-update parent.
    if finished.get('completed_updates') != 1:
        raise ValueError('this continuation entry requires a completed one-update parent')
    recipe = read_json(root/'recipe.json')
    continuation_recipe(recipe['values'],plan)
    recipe_payload={k:v for k,v in recipe.items() if k not in {'fingerprint','unresolved_fields','issues','execution'}}
    if (fingerprint(read_json(root/'config.json')) != parent['config_sha256']
            or recipe['fingerprint'] != parent['recipe_fingerprint']
            or fingerprint(recipe_payload) != parent['recipe_fingerprint']):
        raise ValueError('parent config or recipe identity changed')
    source_root = Path(deployment['parent_source_root'])/'src'/'ricepo'
    sources = {p.name:file_sha256(p) for p in sorted(source_root.glob('*.py'))}
    if sources != read_json(root/'sources.json') or fingerprint(sources) != parent['source_sha256']:
        raise ValueError('parent execution source no longer matches its immutable binding')
    for key in ('dataset_receipt_sha256','manifest_sha256','model_sha256',
                'retriever_sha256','prompt_sha256','behavior'):
        if parent[key] != binding[key]:
            raise ValueError('parent/child asset or behavior differs: '+key)
    old_deployment = read_json(root/'deployment.json')
    if fingerprint({k:v for k,v in old_deployment.items()
                    if k not in {'test_scope','expected_test_domains'}}) != parent['deployment_sha256']:
        raise ValueError('parent deployment identity changed')
    allowed = {'output_root','cache_root','prompt_path','parent_run_root','parent_source_root',
               'continuation_schedule_path','compare_grpo_credit_gradients'}
    for key in old_deployment.keys() | deployment.keys():
        if key not in allowed and old_deployment.get(key) != deployment.get(key):
            raise ValueError('continuation deployment changed: '+key)
    cp = root/'iteration-000000'/'checkpoint'
    saved = verify_checkpoint(cp,bound)
    if (saved['iteration'] != 0 or saved['policy_version'] != finished['last_policy_version']
            or not (cp.parent/'iteration_complete.receipt.json').is_file()):
        raise ValueError('parent is not a completely committed first update')
    step = read_json(cp.parent/'optimizer_step_committed.receipt.json')
    if (step['policy_version'] != saved['policy_version']
            or step['artifacts'].get('update.json') != file_sha256(cp.parent/'update.json')):
        raise ValueError('parent optimizer receipt changed')
    selected=selected_so_far(root,plan['checkpoint_selection'])
    terminal=read_json(cp.parent/'iteration_complete.receipt.json')
    if (not selected or finished.get('selected_checkpoint') != selected
            or terminal.get('policy_version') != saved['policy_version']
            or terminal.get('artifacts',{}).get('selected_checkpoint_so_far') != selected
            or Path(selected['checkpoint']).resolve() != cp.resolve()):
        raise ValueError('parent development selection differs from committed receipts')
    schedule_path = Path(deployment['continuation_schedule_path'])
    schedule = read_json(schedule_path)
    expected = [scheduled_queries(train,plan['training_seeds'][0],i,plan['queries_per_update'])
                for i in range(1,1+plan['total_updates'])]
    if schedule != dict(global_start=1,root_seed=plan['training_seeds'][0],
                        train_manifest_sha256=binding['manifest_sha256']['train'],batches=expected):
        raise ValueError('frozen continuation schedule differs from global seed rule or dataset')
    lineage = dict(parent_binding_sha256=bound,parent_policy_version=saved['policy_version'],
                   parent_checkpoint_complete_sha256=file_sha256(cp/'complete.json'),
                   parent_source_sha256=parent['source_sha256'],parent_run_root=str(root),
                   global_start=1,schedule_sha256=file_sha256(schedule_path),
                   parent_selected_checkpoint=finished['selected_checkpoint'])
    return dict(checkpoint=cp,saved=saved,binding=bound,lineage=lineage,schedule=expected)
