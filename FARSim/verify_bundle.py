"""Verify the delivered file hashes; user result folders are not part of the manifest."""
import json
from pathlib import Path
from run_rate import digest

root = Path(__file__).resolve().parent
expected = json.loads((root/'SHA256SUMS.json').read_text())
bad = []
for name, sha in expected.items():
    p = root/name
    if not p.is_file() or digest(p)!=sha:
        bad.append(name)
if bad:
    raise SystemExit('Missing/changed files: '+', '.join(bad))
print(f'PASS: {len(expected)} files match the delivery manifest.')
