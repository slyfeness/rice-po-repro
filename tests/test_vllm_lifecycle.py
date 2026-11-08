from types import SimpleNamespace
import pytest
from ricepo.vllm_lifecycle import PhaseSeparatedGenerator

def test_policy_change_cannot_reuse_existing_worker():
    g=object.__new__(PhaseSeparatedGenerator)
    g.worker=SimpleNamespace(ready={'policy':'old'})
    g.version=lambda:'new'
    with pytest.raises(RuntimeError,match='stale'):g.prepare_generation()

def test_worker_exits_before_hf_tensors_return_to_gpu():
    events=[];g=object.__new__(PhaseSeparatedGenerator)
    g.worker=SimpleNamespace(close=lambda:events.append('worker_exited'))
    g.actor=SimpleNamespace(to=lambda d:events.append('actor'))
    g.reference=SimpleNamespace(to=lambda d:events.append('reference'))
    g.device='assigned_device';g.moves=[];g.sampler=object()
    g.prepare_scoring()
    assert events==['worker_exited','actor','reference']
    assert g.worker is None and g.sampler is None

def test_failed_shutdown_does_not_overlap_training_residency():
    events=[];g=object.__new__(PhaseSeparatedGenerator)
    def fail():raise RuntimeError('worker still live')
    g.worker=SimpleNamespace(close=fail)
    g.actor=SimpleNamespace(to=lambda d:events.append('actor'))
    with pytest.raises(RuntimeError,match='still live'):g.prepare_scoring()
    assert not events

def test_constructor_failure_reaps_worker(monkeypatch):
    import ricepo.vllm_lifecycle as module
    events=[]
    connection=SimpleNamespace(close=lambda:events.append('pipe_closed'))
    process=SimpleNamespace(start=lambda:events.append('started'))
    context=SimpleNamespace(Pipe=lambda:(connection,connection),Process=lambda **k:process)
    monkeypatch.setattr(module.multiprocessing,'get_context',lambda _:context)
    def fail(self):raise TimeoutError('injected initialization stall')
    monkeypatch.setattr(module.WorkerTransport,'receive',fail)
    monkeypatch.setattr(module.WorkerTransport,'_abort',lambda self:events.append('reaped'),raising=False)
    with pytest.raises(TimeoutError):module.WorkerTransport({})
    assert 'reaped' in events

def test_rpc_namespace_changes_between_resume_attempts(tmp_path):
    from ricepo.vllm_lifecycle import fresh_rpc_path
    assert fresh_rpc_path(tmp_path)!=fresh_rpc_path(tmp_path)

def _sleeping_worker(connection):
    import os,time,subprocess,sys
    os.setsid()
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    connection.send(child.pid)
    time.sleep(60)

def test_abort_reaps_real_owned_process_group():
    import multiprocessing,os,time
    from pathlib import Path
    from ricepo.vllm_lifecycle import WorkerTransport
    if os.name!='posix':pytest.skip('Linux Slurm process group contract')
    ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe()
    proc=ctx.Process(target=_sleeping_worker,args=(child,));proc.start();child.close()
    worker=object.__new__(WorkerTransport);worker.process=proc;worker.connection=parent;worker.closed=False
    try:
        assert parent.poll(15)
        descendant=parent.recv()
        worker._abort()
        assert not proc.is_alive() and proc.exitcode is not None
        path=Path('/proc')/str(descendant)/'stat'
        for _ in range(50):
            if not path.exists() or path.read_text().split()[2]=='Z':break
            time.sleep(.02)
        assert not path.exists() or path.read_text().split()[2]=='Z'
    finally:
        if not worker.closed:worker._abort()
