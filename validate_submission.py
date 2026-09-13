#!/usr/bin/env python3
"""Check a predictions.csv against the required schema. THIS SHIPS TO STUDENTS.

We are testing judgment, not clairvoyance, so nobody should fail on formatting. Run this
before you submit:

    python validate_submission.py predictions.csv

Exit code 0 means the file will be accepted by the grader. Any other exit code prints
exactly what is wrong and where.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys

import pandas as pd

REQUIRED_COLUMNS = ["week_start", "rank", "gateway_id", "score", "reason"]
VISITS_PER_WEEK = 15
SCORED_WEEKS = [dt.date(2026, 2, 2) + dt.timedelta(days=7 * i) for i in range(8)]
MAX_REASON_CHARS = 300

_BARE = re.compile(r"^[0-9A-Fa-f]{12}$")
_COLON = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def normalise_gateway_id(value: str) -> str | None:
    """Accept either id format. The grader normalises too, so pick whichever you prefer."""
    text = str(value).strip()
    if _BARE.match(text):
        return text.upper()
    if _COLON.match(text):
        return text.replace(":", "").upper()
    return None


def validate(path: pathlib.Path) -> list[str]:
    problems: list[str] = []
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # noqa: BLE001 - the message is the point
        return [f"could not read {path} as CSV: {error}"]

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        problems.append(f"missing column(s): {', '.join(missing)}")
    extra = [c for c in frame.columns if c not in REQUIRED_COLUMNS]
    if extra:
        problems.append(f"unexpected column(s): {', '.join(extra)} (exactly {REQUIRED_COLUMNS})")
    if missing:
        return problems  # nothing else is checkable

    expected_rows = VISITS_PER_WEEK * len(SCORED_WEEKS)
    if len(frame) != expected_rows:
        problems.append(
            f"expected {expected_rows} rows ({VISITS_PER_WEEK} per week x "
            f"{len(SCORED_WEEKS)} weeks), found {len(frame)}"
        )

    try:
        weeks = pd.to_datetime(frame["week_start"]).dt.date
    except Exception:  # noqa: BLE001
        problems.append("week_start is not parseable as a date; use YYYY-MM-DD")
        return problems

    found = sorted(set(weeks))
    if found != SCORED_WEEKS:
        unexpected = [str(w) for w in found if w not in SCORED_WEEKS]
        absent = [str(w) for w in SCORED_WEEKS if w not in found]
        if unexpected:
            problems.append(f"week_start values not in the scored window: {unexpected}")
        if absent:
            problems.append(f"missing week_start values: {absent}")

    bad_ids = [v for v in frame["gateway_id"] if normalise_gateway_id(v) is None]
    if bad_ids:
        problems.append(
            f"{len(bad_ids)} gateway_id value(s) are neither 12 hex characters nor "
            f"colon-separated, e.g. {bad_ids[0]!r}"
        )

    if not pd.api.types.is_numeric_dtype(frame["score"]):
        problems.append("score must be numeric")
    elif frame["score"].isna().any():
        problems.append(f"{int(frame['score'].isna().sum())} score value(s) are blank")

    reasons = frame["reason"].astype(str).str.strip()
    if (reasons == "").any() or frame["reason"].isna().any():
        problems.append(f"{int((reasons == '').sum())} reason field(s) are empty")
    too_long = int((reasons.str.len() > MAX_REASON_CHARS).sum())
    if too_long:
        problems.append(f"{too_long} reason field(s) exceed {MAX_REASON_CHARS} characters")

    for week, part in frame.groupby(weeks):
        where = f"week {week}"
        if len(part) != VISITS_PER_WEEK:
            problems.append(f"{where}: {len(part)} rows, expected {VISITS_PER_WEEK}")
        ranks = sorted(pd.to_numeric(part["rank"], errors="coerce").dropna().astype(int))
        if ranks != list(range(1, VISITS_PER_WEEK + 1)):
            problems.append(f"{where}: rank must be 1..{VISITS_PER_WEEK} with no repeats")
        ids = [normalise_gateway_id(v) for v in part["gateway_id"]]
        if len(set(ids)) != len(ids):
            problems.append(f"{where}: the same gateway appears more than once")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=pathlib.Path)
    args = parser.parse_args(argv)

    problems = validate(args.predictions)
    if not problems:
        print(f"{args.predictions}: OK")
        print(f"  {VISITS_PER_WEEK} ranked gateways for each of {len(SCORED_WEEKS)} weeks, "
              f"{SCORED_WEEKS[0]} to {SCORED_WEEKS[-1]}")
        return 0
    print(f"{args.predictions}: {len(problems)} problem(s)\n", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
