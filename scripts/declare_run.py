"""Ships a MIME run as a record: its result (``RunResult.read(FOLDER).to_npz``, MIME's own format) into
``records/data/puncture_runs/NAME/result.npz``, a record for it in ``records.json`` (sha256, size, CC-BY-4.0, credit), and
its entry in the run index (``softtissue.puncture_runs``: the settings it ran at, given here, and the record's name). The
index's own record is updated to its new bytes. Run from the plugin's root, in an environment with the plugin installed:

    python scripts/declare_run.py OUT_FOLDER NAME tip=cone material=softtissue.whole_blood_clot dimension=3d h_mm=0.25 ...

A setting that reads as a number is stored as one. A run of that name already in the index is replaced, with its record."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from mime.results import RunResult

RECORDS = Path("src/mime_softtissue/records")
DATA = RECORDS / "data" / "puncture_runs"
CREDIT = "computed by the plugin's authors with MIME (examples/puncture.py, GitHub-hosted runners, float64)"


def number_or_text(text: str) -> object:
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def declared(name: str, path: Path, credit: str) -> dict[str, object]:
    body = path.read_bytes()
    return {"name": name, "sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "licence": "CC-BY-4.0",
            "locations": [f"package:softtissue/{path.relative_to(RECORDS).as_posix()}"], "credit": credit,
            "role": "results", "kind": "authored"}


def main() -> None:
    folder, name, *pairs = sys.argv[1:]
    settings = {k: number_or_text(v) for k, v in (p.split("=", 1) for p in pairs)}
    result = RunResult.read(folder)
    out = DATA / name / "result.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_npz(out)
    RunResult.read(out)  # reads back, or refuses here
    record = f"softtissue.puncture_run.{name}.result"
    index_path = DATA / "index.json"
    index = json.loads(index_path.read_text())
    index["runs"][name] = {**settings, "result": record}
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    manifest_path = RECORDS / "records.json"
    manifest = json.loads(manifest_path.read_text())
    kept = [r for r in manifest["assets"] if r["name"] not in (record, "softtissue.puncture_runs")]
    index_credit = next(r["credit"] for r in manifest["assets"] if r["name"] == "softtissue.puncture_runs")
    manifest["assets"] = kept + [declared(record, out, CREDIT), declared("softtissue.puncture_runs", index_path,
                                                                         index_credit)]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{name}: {out} ({out.stat().st_size} bytes), record {record}; the index lists {len(index['runs'])} runs")


if __name__ == "__main__":
    main()
