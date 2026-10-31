"""Repository-relative, read-only inputs and separate generated outputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def new_output(path):
    path=(ROOT/Path(path)).resolve()
    if not path.is_relative_to(ROOT/'outputs') or path==ROOT/'outputs':
        raise ValueError('Generated output must be a new child directory of outputs/')
    if path.exists():raise FileExistsError('Existing output is protected: '+str(path))
    path.mkdir(parents=True)
    return path

def check_inputs():
    manifest=json.loads((ROOT/'data/input_manifest.json').read_text())
    for relative,digest in manifest['input_hashes'].items():
        path=ROOT/relative
        if not path.is_file() or sha(path)!=digest:
            raise ValueError('Missing or changed frozen input: '+relative)
    return len(manifest['input_hashes'])

def json_write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
