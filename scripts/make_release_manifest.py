#!/usr/bin/env python3
"""Write release-manifest.json and SHA256SUMS for the files tracked by git.

Run from a git checkout after staging the release files:

    git add -A && python3 scripts/make_release_manifest.py

Both files list the distributed bytes. The manifest and the checksum list do not hash
themselves; SHA256SUMS also lists the manifest. Producer digests stored inside scientific
records identify the computation that produced them and are not changed by this script.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"release-manifest.json", "SHA256SUMS"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checkout = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if checkout.returncode == 0 and Path(checkout.stdout.strip()).resolve() == ROOT and listing.returncode == 0:
        listed = listing.stdout.decode().split("\0")
    else:
        # The assembled archive is an isolated distribution. Never accidentally
        # bind it to an enclosing author's development checkout.
        listed = sorted(str(path.relative_to(ROOT)) for path in ROOT.rglob('*')
                        if path.is_file() and not any(part in {'.git', '__pycache__', 'tmp'}
                                                     for part in path.relative_to(ROOT).parts)
                        and path.suffix != '.pyc')
    files = sorted(name for name in listed if name and name not in SKIP)
    info = json.loads((ROOT / "release-info.json").read_text())
    entries = {name: {"sha256": digest(ROOT / name), "bytes": (ROOT / name).stat().st_size,
                      "distribution": "versioned-archive"} for name in files}
    manifest = {
        "schema_version": 1, "study": "reference-anchored-geometry", "version": info["version"],
        "publication_status": info.get('publication_status', 'unspecified'),
        "scope": "Distributed source and research files. Excludes this manifest and SHA256SUMS to avoid self-reference.",
        "files": entries, "source_files": len(entries), "checkpoint_files": 0,
        "historical_hash_policy": "Producer digests inside scientific records identify historical computations. This manifest hashes the actual distributed bytes.",
    }
    (ROOT / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [f"{entries[name]['sha256']}  {name}" for name in files]
    lines.append(f"{digest(ROOT / 'release-manifest.json')}  release-manifest.json")
    (ROOT / "SHA256SUMS").write_text("\n".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])) + "\n")
    print(f"{len(entries)} files hashed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
