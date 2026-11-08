"""Verified local staging before shared checkpoint publication; no optimizer step."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from .artifacts import file_sha256,atomic_json

def state_digest(value):
    """Digest logical optimizer/RNG tensors before and after serialization."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(json.dumps([str(v.dtype),list(v.shape)]).encode())
            raw=v.detach().contiguous().reshape(-1).view(torch.uint8)
            for block in raw.split(4*1024*1024):h.update(block.cpu().numpy().tobytes())
        elif isinstance(v,bytes):
            h.update(b'bytes:');h.update(str(len(v)).encode());h.update(b':');h.update(v)
        elif isinstance(v,dict):
            for key in sorted(v,key=lambda k:(type(k).__name__,str(k))):
                h.update(json.dumps([type(key).__name__,str(key)]).encode());visit(v[key])
        elif isinstance(v,(tuple,list)):
            h.update(str(type(v).__name__).encode());h.update(str(len(v)).encode())
            for item in v:visit(item)
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    visit(value);return h.hexdigest()

def rng_probe(payload,device):
    import torch
    device=torch.device(device)
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        torch.random.set_rng_state(payload['cpu_rng'])
        cpu=torch.rand(32).numpy().tobytes()
        result=dict(cpu_draw_sha256=hashlib.sha256(cpu).hexdigest())
        if device.type=='cuda':
            torch.cuda.set_rng_state(payload['cuda_rng'],device)
            gpu=torch.rand(32,device=device).cpu().numpy().tobytes()
            result['cuda_draw_sha256']=hashlib.sha256(gpu).hexdigest()
    return result

def verified_local_copy(destination,write,verify,*,temporary_root=None):
    """Caller publishes only after this returns. A failed staging dir is retained."""
    destination=Path(destination)
    parent=Path(temporary_root or os.environ.get('SLURM_TMPDIR') or '/tmp').resolve()
    local=Path(tempfile.mkdtemp(prefix='ricepo-checkpoint-',dir=parent))
    try:
        write(local);verify(local)
        files={str(p.relative_to(local)):file_sha256(p) for p in local.rglob('*') if p.is_file()}
        if not files:raise ValueError('empty checkpoint staging')
        for name,digest in files.items():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with (local/name).open('rb') as src,target.open('wb') as dst:
                shutil.copyfileobj(src,dst,4*1024*1024);dst.flush();os.fsync(dst.fileno())
            if file_sha256(target)!=digest:raise IOError('shared checkpoint copy checksum mismatch: '+name)
        # Local verification plus exact shared-copy hashes precedes atomic publication.
        atomic_json(destination/'io-verification.json',dict(mode='local_verified_v1',files=files,
            local_payload_verified=True,shared_copy_verified=True))
    except Exception as e:
        atomic_json(destination/'io-failure.json',dict(type=type(e).__name__,reason=str(e),
            retained_local_staging=str(local),published=False))
        raise
    else:
        if local.resolve().parent!=parent or not local.name.startswith('ricepo-checkpoint-'):
            raise ValueError('refuse cleanup outside created checkpoint staging')
        shutil.rmtree(local)
