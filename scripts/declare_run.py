"""Ships a MIME run as a record: its result (``RunResult.read(FOLDER).to_npz``, MIME's own format) into
``records/data/puncture_runs/NAME/result.npz``, a record for it in ``records.json`` (sha256, size, CC-BY-4.0, credit), and
its entry in the run index (``softtissue.puncture_runs``: the settings it ran at, given here, and the record's name). The
index's own record is updated to its new bytes. Run from the plugin's root, in an environment with the plugin installed:

    python scripts/declare_run.py OUT_FOLDER NAME tip=cone material=softtissue.whole_blood_clot dimension=3d h_mm=0.25 ...

A setting that reads as a number is stored as one. A run of that name already in the index is replaced, with its record.

The 3-D engine's own runs (``PunctureRun.save`` wrote ``PREFIX.npz`` and ``PREFIX.json``) ship the same way, as two
records (``softtissue.puncture3d.NAME.rows`` and ``.meta``) and an entry in the index's ``engine_runs``:

    python scripts/declare_run.py --engine PREFIX NAME"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from mime.results import RunResult

RECORDS = Path("src/mime_softtissue/records")
DATA = RECORDS / "data" / "puncture_runs"
CREDIT = "computed by the plugin's authors with MIME (examples/puncture.py, GitHub-hosted runners, float64)"
ENGINE_CREDIT = ("computed by the plugin's authors with the plugin's 3-D engine (engine.puncture3d, outside MIME, "
                 "GitHub-hosted runners, float64)")


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


def update(index: dict[str, object], records: list[dict[str, object]], replaced: set[str]) -> None:
    """Writes the index, then the manifest with ``records`` (and the index's own record) in place of ``replaced``."""
    index_path = DATA / "index.json"
    index_path.write_text(json.dumps(index, indent=1) + "\n")
    manifest_path = RECORDS / "records.json"
    manifest = json.loads(manifest_path.read_text())
    index_credit = next(r["credit"] for r in manifest["assets"] if r["name"] == "softtissue.puncture_runs")
    kept = [r for r in manifest["assets"] if r["name"] not in replaced | {"softtissue.puncture_runs"}]
    manifest["assets"] = kept + records + [declared("softtissue.puncture_runs", index_path, index_credit)]
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")


def engine(prefix: str, name: str) -> None:
    import shutil
    target = RECORDS / "data" / "puncture3d"
    target.mkdir(parents=True, exist_ok=True)
    made = []
    for ext, part in ((".npz", "rows"), (".json", "meta")):
        path = target / (name + ext)
        if Path(prefix + ext).resolve() != path.resolve():
            shutil.copyfile(prefix + ext, path)
        made.append(declared(f"softtissue.puncture3d.{name}.{part}", path, ENGINE_CREDIT))
    index = json.loads((DATA / "index.json").read_text())
    index["engine_runs"][name] = {"rows": f"softtissue.puncture3d.{name}.rows", "meta": f"softtissue.puncture3d.{name}.meta"}
    update(index, made, {str(r["name"]) for r in made})
    print(f"{name}: two records in {target}; the index lists {len(index['engine_runs'])} engine runs")


def main() -> None:
    if sys.argv[1] == "--engine":
        engine(sys.argv[2], sys.argv[3])
        return
    folder, name, *pairs = sys.argv[1:]
    settings = {k: number_or_text(v) for k, v in (p.split("=", 1) for p in pairs)}
    result = RunResult.read(folder)
    out = DATA / name / "result.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_npz(out)
    RunResult.read(out)  # reads back, or refuses here
    record = f"softtissue.puncture_run.{name}.result"
    index = json.loads((DATA / "index.json").read_text())
    index["runs"][name] = {**settings, "result": record}
    update(index, [declared(record, out, CREDIT)], {record})
    print(f"{name}: {out} ({out.stat().st_size} bytes), record {record}; the index lists {len(index['runs'])} runs")


if __name__ == "__main__":
    main()
