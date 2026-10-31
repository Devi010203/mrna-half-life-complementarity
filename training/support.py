"""Small I/O and validation dependencies for the optional scientific implementations."""
import hashlib
import json
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def safe(root, relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path.is_relative_to(root), 'Output escapes the chosen training directory')
    require(root.is_relative_to(Path(__file__).resolve().parents[1]/'outputs'), 'Training writes require outputs/')
    return path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def js(root, relative, value):
    path=safe(root,relative);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2)

def csv(root, relative, frame):
    path=safe(root,relative);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:frame.to_csv(f,index=False)
