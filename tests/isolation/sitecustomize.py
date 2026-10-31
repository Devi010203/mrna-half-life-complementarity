"""Opt-in clean-run guard, activated by the release test launcher only."""
import os
from pathlib import Path
import sys

root_value=os.environ.get('MRNA_ISOLATED_ROOT')
if root_value:
    root=Path(root_value).resolve()
    # No historical project directory is readable; interpreter/system libraries
    # are allowed. strace independently records native-extension file accesses.
    denied_parent=Path(os.environ['MRNA_FORBIDDEN_PARENT']).resolve()
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(denied_parent) and not path.is_relative_to(root):
                raise PermissionError('Clean-run guard: access outside the release is forbidden')
            mode=args[1] if len(args)>1 else None
            flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not path.is_relative_to(root) and str(path) not in {'/dev/null','/dev/tty'}:
                raise PermissionError('Clean-run guard: output outside the release is forbidden')
        if event=='socket.connect':raise PermissionError('Clean-run evaluation must be offline')
    sys.addaudithook(guard)
