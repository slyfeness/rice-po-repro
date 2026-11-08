"""Strict qualification gate for the observed 128-query HF memory repair."""
import ast,hashlib,os
from pathlib import Path
from .artifacts import file_sha256,require_cluster_storage
from .state import read_json

MODE='cpu_saved_expandable_deferred_adam_v1'


def normalized_source_sha(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf8').encode('utf8')).hexdigest()


def residency_methods_sha(path):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PhaseSeparatedGenerator')
    names={'_offload','_prepare_scoring','prepare_optimizer_step'}
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(nodes)!=len(names):raise ValueError('incomplete residency implementation')
    return hashlib.sha256('\n'.join(ast.dump(n,include_attributes=False) for n in nodes).encode()).hexdigest()


def validate_memory_repair(spec,sampling_plan_sha256):
    if not isinstance(spec,dict) or set(spec)!={'mode','qualification_result_path','qualification_result_sha256'} or spec['mode']!=MODE:
        raise ValueError('unsupported HF memory repair configuration')
    path=require_cluster_storage(spec['qualification_result_path']);record=read_json(path)
    if file_sha256(path)!=spec['qualification_result_sha256'] or record.get('status')!='PASS' or record.get('mode')!=MODE:
        raise ValueError('unqualified HF memory repair')
    source=Path(__file__).parent
    if (record['sampling_plan_sha256']!=sampling_plan_sha256
            or record['modeling_normalized_source_sha256']!=normalized_source_sha(source/'modeling.py')
            or record['optimizer_residency_methods_sha256']!=residency_methods_sha(source/'vllm_lifecycle.py')):
        raise ValueError('memory qualification and actual implementation differ')
    if os.environ.get('PYTORCH_ALLOC_CONF')!='expandable_segments:True':
        raise ValueError('HF allocator repair must be explicitly configured before Torch initialization')
    return record
