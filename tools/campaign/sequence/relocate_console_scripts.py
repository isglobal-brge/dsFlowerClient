#!/usr/bin/env python3
"""Update copied venv entrypoint interpreter paths, preserving Python bodies."""
import hashlib
import json
from pathlib import Path
import sys

venv = Path(sys.argv[1]).resolve()
assert Path("/opt/cells-sequence") in venv.parents
records = []
for path in sorted((venv / "bin").iterdir()):
    if path.is_symlink() or not path.is_file():
        continue
    with path.open("rb") as stream:
        first = stream.readline()
        if not first.startswith(b"#!/workspace/cells-sequence/") or b"/bin/python" not in first:
            continue
        body = stream.read()
    replacement = f"#!{venv}/bin/python\n".encode()
    path.write_bytes(replacement + body)
    path.chmod(0o755)
    assert path.read_bytes().split(b"\n", 1)[1] == body
    records.append({"script": path.name, "old_interpreter": first.decode().strip(),
                    "new_interpreter": replacement.decode().strip(),
                    "unchanged_body_sha256": hashlib.sha256(body).hexdigest()})
result = {"venv": str(venv), "python_bodies_unchanged": True, "entrypoints": records}
(venv / "sequence-console-relocation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
