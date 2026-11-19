from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from daph_ext.probes.sandbox import DockerSandboxConfig, sandbox_environment_fingerprint

CASES = {
    "network_connect": """import socket, pytest\ndef test_blocked():\n    s=socket.socket(); s.settimeout(1)\n    with pytest.raises(OSError): s.connect(('1.1.1.1', 53))\n""",
    "write_readonly_repo": """from pathlib import Path\ndef test_ro_repo():\n    p=Path('/repo/SHOULD_NOT_WRITE')\n    try: p.write_text('x')\n    except OSError: return\n    raise AssertionError('read-only repo was writable')\n""",
    "write_outside_workspace": """from pathlib import Path\ndef test_ro_root():\n    p=Path('/SHOULD_NOT_WRITE')\n    try: p.write_text('x')\n    except OSError: return\n    raise AssertionError('read-only root was writable')\n""",
    "read_host_path": """from pathlib import Path\ndef test_no_host_secret():\n    assert not Path('/host').exists()\n""",
    "privileged_syscall": """import os, pytest\ndef test_no_setuid_root():\n    with pytest.raises(PermissionError): os.setuid(0)\n""",
    "mount_attempt": """import subprocess\ndef test_mount_blocked():\n    p=subprocess.run(['mount','-t','tmpfs','tmpfs','/mnt'],capture_output=True)\n    assert p.returncode != 0\n""",
}


def _run_container(repo: Path, cfg: DockerSandboxConfig) -> tuple[int, str]:
    cmd = [
        'docker','run','--rm','--pull=never','--network',cfg.network,
        '--cpus',str(cfg.cpus),'--memory',cfg.memory,'--memory-swap',cfg.memory,
        '--pids-limit',str(cfg.pids_limit),'--ulimit','nofile=1024:1024',
        '--security-opt','no-new-privileges:true','--cap-drop','ALL',
        '--user','65534:65534','--ipc','none',
        '--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=512m',
        '--tmpfs','/workspace:rw,nosuid,nodev,size=2g',
        '-v',f'{repo.resolve()}:/repo:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','HOME=/tmp',
    ]
    if cfg.read_only_root:
        cmd.append('--read-only')
    shell = 'cp -a /repo/. /workspace/ && cd /workspace && python -m pytest -q -p no:cacheprovider'
    cmd += [cfg.image,'/bin/sh','-lc',shell]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_seconds, check=False)
        return p.returncode, (p.stdout or '') + '\n' + (p.stderr or '')
    except subprocess.TimeoutExpired:
        return 124, 'timeout'


def _resource_limit_cases(cfg: DockerSandboxConfig) -> dict[str, bool]:
    # These are container-level checks rather than pytest assertions because successful
    # containment may terminate the process/container.
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as td:
        repo=Path(td); (repo/'test_dummy.py').write_text('def test_ok(): assert True\n')
        base=['docker','run','--rm','--pull=never','--network','none','--cap-drop','ALL','--security-opt','no-new-privileges:true','--user','65534:65534']
        # PID/fork pressure: command must not run indefinitely or escape the pids limit.
        cmd=base+['--pids-limit','32','--memory','256m','--cpus','0.5',cfg.image,'/bin/sh','-lc','i=0; while [ $i -lt 200 ]; do sleep 2 & i=$((i+1)); done; wait']
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=5,check=False)
            out['pid_limit']=p.returncode != 0
        except subprocess.TimeoutExpired:
            out['pid_limit']=False
        # Timeout enforcement is owned by host runner: a 30s sleep must be killed by 1s timeout.
        cmd=base+[cfg.image,'/bin/sh','-lc','sleep 30']
        try:
            subprocess.run(cmd,capture_output=True,text=True,timeout=1,check=False)
            out['timeout_enforcement']=False
        except subprocess.TimeoutExpired:
            out['timeout_enforcement']=True
        # Memory pressure should terminate/fail under a small hard limit.
        cmd=base+['--memory','128m','--memory-swap','128m',cfg.image,'python','-c','x=bytearray(1024*1024*512); print(len(x))']
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=10,check=False)
        out['memory_limit']=p.returncode != 0
        out['fork_bomb_guard']=out['pid_limit']
    return out


def main() -> None:
    ap=argparse.ArgumentParser(description='Adversarial Docker sandbox qualification')
    ap.add_argument('--image', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--timeout-seconds', type=int, default=30)
    args=ap.parse_args()
    cfg=DockerSandboxConfig(image=args.image, timeout_seconds=args.timeout_seconds)
    fingerprint=sandbox_environment_fingerprint(args.image)
    case_results: dict[str,bool]={}
    with tempfile.TemporaryDirectory() as td:
        repo=Path(td)
        for name,src in CASES.items():
            (repo/f'test_{name}.py').write_text(src)
        rc, combined=_run_container(repo,cfg)
        # all individual assertions must pass; rc 0 means every containment assertion held.
        for name in CASES: case_results[name]=(rc==0)
        if rc != 0:
            case_results['pytest_container_clean_exit']=False
        else:
            case_results['pytest_container_clean_exit']=True
    case_results.update(_resource_limit_cases(cfg))
    required={'network_connect','write_readonly_repo','write_outside_workspace','read_host_path','fork_bomb_guard','memory_limit','pid_limit','privileged_syscall','mount_attempt','timeout_enforcement'}
    passed=all(case_results.get(k) is True for k in required)
    payload={'data_origin':'empirical','experiment_name':'sandbox_qualification','image':args.image,'image_fingerprint':fingerprint,'cases':case_results,'required_cases':sorted(required),'passed':passed}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if not passed: raise SystemExit('sandbox qualification FAILED')
    print('sandbox qualification PASS')

if __name__=='__main__': main()
