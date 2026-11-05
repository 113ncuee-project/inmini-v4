"""Bounded cohort admission compatible with the existing serial PPA lock.

Each managed parent holds a shared lock on the SAME global byte. Unmanaged
old/new experiments still require its exclusive lock and cannot overlap a
cohort, even if the coordinator exits while its case parents are still alive.
Separate exclusive slot locks bound admitted cases. This changes host-side
execution only, never the physical model, workload or a policy trajectory.

Windows semantics: https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex
Shared locks overlap; the legacy msvcrt exclusive lock is blocked by them.
"""
from contextlib import contextmanager,ExitStack
import ctypes
import json
import os
from pathlib import Path

from run_local_rl_diagnostics import lock,sources,sha


@contextmanager
def shared_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            from ctypes import wintypes
            class Overlapped(ctypes.Structure):
                _fields_ = [('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
                            ('Offset',wintypes.DWORD),('OffsetHigh',wintypes.DWORD),('hEvent',wintypes.HANDLE)]
            kernel = ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.LockFileEx.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD,
                                         wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(Overlapped)]
            kernel.LockFileEx.restype = wintypes.BOOL
            kernel.UnlockFileEx.argtypes = [wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD,
                                           wintypes.DWORD,ctypes.POINTER(Overlapped)]
            kernel.UnlockFileEx.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(stream.fileno())
            overlap = Overlapped()
            if not kernel.LockFileEx(handle,1,0,1,0,ctypes.byref(overlap)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield
            finally:
                if not kernel.UnlockFileEx(handle,0,1,0,ctypes.byref(overlap)):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_SH|fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream,fcntl.LOCK_UN)


def validate_lease(workspace,output,lease_path,slot,source_identity):
    workspace,output,lease_path = Path(workspace).resolve(),Path(output).resolve(),Path(lease_path).resolve()
    if not lease_path.is_relative_to(workspace/'output'):
        raise ValueError('Cohort lease must remain in project output')
    lease = json.loads(lease_path.read_text(encoding='utf-8'))
    if lease.get('status') != 'active' or lease.get('workspace') != str(workspace) or lease.get('source_identity') != source_identity:
        raise ValueError('Inactive or mismatched cohort lease')
    limit = lease.get('jobs')
    if type(limit) is not int or not 1 <= limit <= 4 or type(slot) is not int or not 0 <= slot < limit:
        raise ValueError('Cohort slot outside declared1..4 capacity')
    root = Path(lease['output_root']).resolve()
    if not root.is_relative_to(workspace/'output') or not output.is_relative_to(root):
        raise ValueError('Case output escapes cohort')
    if str(output) not in lease.get('allowed_outputs',[]):
        raise ValueError('Case was not declared by the coordinator')
    return lease


@contextmanager
def admission(workspace,output,app,lease_path=None,slot=0):
    workspace = Path(workspace).resolve()
    global_path = workspace/'output/.local_rl_diagnostics.lock'
    if not lease_path:
        with lock(global_path):
            yield
        return
    path = Path(lease_path).resolve()
    validate_lease(workspace,output,path,slot,sha(sources(Path(app))))
    with ExitStack() as stack:
        stack.enter_context(shared_lock(global_path))
        stack.enter_context(lock(path.parent/'slots'/f'slot{slot}.lock'))
        # Check after both locks, closing an admission/end-of-cohort race.
        validate_lease(workspace,output,path,slot,sha(sources(Path(app))))
        yield
