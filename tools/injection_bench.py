#!/usr/bin/env python3
"""Score the injection detector against the corpus.

    packages/brain/.venv/bin/python tools/injection_bench.py
    packages/brain/.venv/bin/python tools/injection_bench.py --verbose
    packages/brain/.venv/bin/python tools/injection_bench.py --sweep

Two numbers matter and they trade against each other:

- **Recall** — the share of attacks flagged. A miss is an attack that keeps a
  standing approval alive.
- **False-positive rate** — the share of benign content flagged. Every false
  positive costs the user an approval prompt they did not need, and enough of
  them train the reflex this whole layer exists to protect: approving without
  reading.

``--sweep`` prints both across every threshold so the chosen one is a decision
with a table behind it rather than a preference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "brain" / "src"))

from june_brain.guard import injection  # noqa: E402

CORPUS = ROOT / "packages" / "brain" / "tests" / "fixtures" / "injection_corpus"


@dataclass
class Case:
    id: str
    category: str
    text: str
    notes: str


def load(name: str) -> list[Case]:
    data = json.loads((CORPUS / name).read_text(encoding="utf-8"))
    return [
        Case(c["id"], c["category"], c["text"], c.get("notes", ""))
        for c in data["cases"]
    ]


def sweep(attacks: list[Case], benign: list[Case]) -> None:
    print("\nThreshold sweep\n")
    print(f"{'min_score':>10} {'recall':>9} {'FP rate':>9}  {'':>9}")
    print("-" * 42)
    for min_score in range(1, 9):
        hits = sum(injection.scan(c.text).score >= min_score for c in attacks)
        fps = sum(injection.scan(c.text).score >= min_score for c in benign)
        chosen = min_score == injection.SUSPICIOUS_SCORE
        print(
            f"{min_score:>10} {hits / len(attacks):>8.0%} {fps / len(benign):>9.0%}"
            f"  {'<- chosen' if chosen else '':>9}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list every case")
    ap.add_argument("--sweep", action="store_true", help="print the threshold table")
    args = ap.parse_args()

    all_attacks, benign = load("attacks.json"), load("benign.json")

    # A multi_stage case is only an attack alongside its partner — scoring the
    # halves individually would report a miss for behaviour that is correct.
    # They are scored below through scan_all, which is what dispatch uses.
    staged = [c for c in all_attacks if c.category == "multi_stage"]
    attacks = [c for c in all_attacks if c.category != "multi_stage"]

    misses: list[tuple[Case, injection.InjectionScan]] = []
    for case in attacks:
        result = injection.scan(case.text)
        if not result.suspicious:
            misses.append((case, result))
        elif args.verbose:
            print(f"  hit  {case.id:<26} {result.score:>2}  {','.join(result.signals)}")

    false_positives: list[tuple[Case, injection.InjectionScan]] = []
    for case in benign:
        result = injection.scan(case.text)
        if result.suspicious:
            false_positives.append((case, result))
        elif args.verbose and result.signals:
            print(f"  near {case.id:<26} {result.score:>2}  {','.join(result.signals)}")

    staged_alone = all(not injection.scan(c.text).suspicious for c in staged)
    staged_caught = injection.scan_all([c.text for c in staged]).suspicious if staged else True

    start = time.perf_counter()
    blob = "\n".join(c.text for c in (*attacks, *benign))[: injection.MAX_SCAN_CHARS]
    for _ in range(100):
        injection.scan(blob)
    per_scan_ms = (time.perf_counter() - start) / 100 * 1000

    recall = (len(attacks) - len(misses)) / len(attacks)
    fp_rate = len(false_positives) / len(benign)

    print(f"\nAttacks   {len(attacks):>3}   caught {len(attacks) - len(misses):>3}   recall {recall:.0%}")
    print(f"Benign    {len(benign):>3}   flagged {len(false_positives):>3}   FP rate {fp_rate:.0%}")
    print(f"Staged    {len(staged):>3}   quiet alone {staged_alone}, caught together {staged_caught}")
    print(f"Scan cost {per_scan_ms:.2f}ms for {len(blob)} chars")

    if misses:
        print(f"\nMissed ({len(misses)}):")
        for case, result in misses:
            print(f"  {case.id:<26} score {result.score}  [{','.join(result.signals) or 'nothing'}]")
            print(f"    {case.notes}")
    if false_positives:
        print(f"\nFalse positives ({len(false_positives)}):")
        for case, result in false_positives:
            print(f"  {case.id:<26} score {result.score}  [{','.join(result.signals)}]")
            print(f"    {case.notes}")

    if args.sweep:
        sweep(attacks, benign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
