"""Negative control: this intentionally attempts a forbidden project read."""
import os
from pathlib import Path

candidate=Path(os.environ['MRNA_FORBIDDEN_PARENT'])/'for_github.md'
try:
    candidate.read_bytes()
except PermissionError:
    print('PASS: read of a non-release project file was blocked')
else:
    raise AssertionError('Isolation guard was not active')
