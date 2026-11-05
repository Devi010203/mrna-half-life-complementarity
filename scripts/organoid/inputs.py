"""Portable integrity and output boundaries; contains no statistical functions."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_inputs():
    manifest=json.loads((ROOT/'data/organoid/input_manifest.json').read_text())
    for path,want in manifest['input_hashes'].items():
        if digest(ROOT/path)!=want:
            raise ValueError('Changed organoid input/reference: '+path)
    frozen=ROOT/'data/organoid/frozen'
    record=json.loads((frozen/'evaluation_v2_20261102/PROTOCOL_V2_FREEZE.json').read_text())
    for row in record['files']:
        if digest(frozen/row['path'])!=row['sha256']:
            raise ValueError('Original protocol freeze mismatch: '+row['path'])
    return len(manifest['input_hashes'])

def output_path(path,existing=False):
    target=(ROOT/path).resolve()
    if not target.is_relative_to(ROOT/'outputs') or target==ROOT/'outputs':
        raise ValueError('Use a child of repository outputs/')
    if existing:
        if not target.is_dir(): raise FileNotFoundError(target)
    elif target.exists():
        raise FileExistsError('Refusing to overwrite an existing run')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
    return target
