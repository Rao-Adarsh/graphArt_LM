"""Build the small probe set for RQ6 (a construct held out of training).

    python scripts/build_rq6_probe.py

Writes data/generated/rq6.jsonl: everything needed to see whether an adapter trained
without any dead branch (`train_lora.py --exclude-construct dead_branch`) can handle
one, and a control to show it is the construct, not the smaller training set:

  dead_branch   tier-3 test programs with an `if self.<flag>:` whose flag is False (18)
  control       the other tier-3 test programs, without one (22) -- same tier, same generator
  dead_flag     edited programs with a dead branch added (from the edit set)
  branch_on     edited programs whose dead branch was switched on (from the edit set)

Small on purpose: ~80 programs, ~20 minutes on a T4.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_lora import has_dead_branch

OUT = ROOT / "data" / "generated" / "rq6.jsonl"


def main() -> None:
    test = [json.loads(l) for l in (ROOT / "data/generated/test.jsonl").open(encoding="utf-8")]
    edits = [json.loads(l) for l in (ROOT / "data/generated/edits.jsonl").open(encoding="utf-8")]
    rows = [{**r, "group": "dead_branch" if has_dead_branch(r) else "control"}
            for r in test if r["tier"] == 3]
    rows += [{**r, "group": r["edit"]} for r in edits if r["edit"] in ("dead_flag", "branch_on")]
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    counts = {}
    for r in rows:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(f"{len(rows)} programs -> {OUT.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
