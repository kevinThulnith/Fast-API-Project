"""
comparison_report.py

Compares the "traditional" testing pipeline (hand-written tests/test.py +
LLM-generated-but-fully-run tests_generated/generated_test.py) against the
"agentic" pipeline (TestSelectAgent's predictively-selected subset).

Reads the artifacts your CI/CD workflow already produces:

  Traditional:
    - first_test.csv              (tests/test.py run,            pytest-csv)
    - first_test_coverage.json    (tests/test.py coverage,       coverage.py json)
    - second_test.csv             (generated_test.py run,        pytest-csv)
    - second_test_coverage.json   (generated_test.py coverage,   coverage.py json)

  Agentic:
    - selecter_results.csv        (TestSelectAgent's run,        pytest-csv, via _write_csv)
    - coverage_selector.json      (TestSelectAgent's coverage,   coverage.py json)

Writes one CSV report covering pass/fail counts, coverage %, and the
manual-effort proxies from the research proposal (lines of test code,
number of test functions) for each side, plus the deltas between them.

Any input file that is missing (e.g. the selector ran with zero tests
selected and so never produced a coverage file) is treated as "no data"
rather than crashing the script — every metric row for that side is left
blank and a warning is printed, so the CI artifact still gets written even
on a partial run.

Usage:
    python comparison_report.py \\
        --first-test-csv first_test.csv \\
        --first-test-coverage first_test_coverage.json \\
        --second-test-csv second_test.csv \\
        --second-test-coverage second_test_coverage.json \\
        --selector-csv selecter_results.csv \\
        --selector-coverage coverage_selector.json \\
        --out reports/comparison_report.csv
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Test-file line/function counting (manual-effort proxy)
# ---------------------------------------------------------------------------
TEST_FUNC_PATTERN = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+", re.MULTILINE)


def count_test_file_stats(path: str) -> Optional[Dict[str, int]]:
    """Return {lines, test_functions} for a test file, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.count("\n") + (
            1 if content and not content.endswith("\n") else 0
        )
        test_functions = len(TEST_FUNC_PATTERN.findall(content))
        return {"lines": lines, "test_functions": test_functions}
    except Exception as e:
        print(f"⚠️  Could not read test file {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# pytest-csv reading
#
# Schema (from --csv-columns=id,status,duration,message), confirmed against
# a real pytest-csv run: header row, then one row per test with status one
# of "passed" / "failed" / "skipped" / "error". Empty file body (header
# only) is valid and means zero tests ran.
# ---------------------------------------------------------------------------
def read_pytest_csv(path: str) -> Optional[Dict[str, object]]:
    """Return {total, passed, failed, skipped, error, tests: [...]} or None if missing."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"⚠️  Could not read CSV {path}: {e}")
        return None

    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for row in rows:
        status = (row.get("status") or "").strip().lower()
        if status in counts:
            counts[status] += 1
    return {
        "total": len(rows),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "error": counts["error"],
        "tests": rows,
    }


# ---------------------------------------------------------------------------
# coverage.py JSON reading
#
# Schema confirmed against a real `coverage json` run: top-level keys
# "meta", "files", "totals". The fields used below
# (percent_covered, covered_lines, num_statements, missing_lines) are part
# of coverage.py's stable JSON report format.
# ---------------------------------------------------------------------------
def read_coverage_json(path: str) -> Optional[Dict[str, float]]:
    """Return the relevant totals from a coverage.py JSON report, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not read coverage JSON {path}: {e}")
        return None

    totals = data.get("totals", {})
    if not totals:
        print(f"⚠️  Coverage JSON {path} has no 'totals' section.")
        return None
    return {
        "percent_covered": round(totals.get("percent_covered", 0.0), 2),
        "covered_lines": totals.get("covered_lines", 0),
        "num_statements": totals.get("num_statements", 0),
        "missing_lines": totals.get("missing_lines", 0),
    }


# ---------------------------------------------------------------------------
# Combine two CSV-run summaries (e.g. first_test + second_test = "traditional")
# ---------------------------------------------------------------------------
def combine_run_summaries(
    summaries: List[Optional[Dict[str, object]]],
) -> Optional[Dict[str, object]]:
    present = [s for s in summaries if s is not None]
    if not present:
        return None
    combined = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for s in present:
        for key in ("total", "passed", "failed", "skipped", "error"):
            combined[key] += s.get(key, 0)
    return combined


def combine_coverage(
    coverages: List[Optional[Dict[str, float]]],
) -> Optional[Dict[str, float]]:
    """
    Combine multiple coverage.py JSON totals into one statement-weighted
    percentage. Summing num_statements/covered_lines from separate reports
    on the SAME file (main.py, in this project) double-counts the
    denominator if both runs cover the same module — so this assumes the
    two reports are for two different full runs against the same target
    and reports their UNION via line counts is not attempted. Instead we
    report a simple statement-count-weighted average, which is correct
    when each report's num_statements reflects the same target module.
    """
    present = [c for c in coverages if c is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    # Weighted average by num_statements (in this project all reports
    # target the same single module, main.py, so num_statements should
    # match across reports — weighting still degrades gracefully if not).
    total_weight = sum(c["num_statements"] for c in present) or 1
    weighted_percent = (
        sum(c["percent_covered"] * c["num_statements"] for c in present) / total_weight
    )
    return {
        "percent_covered": round(weighted_percent, 2),
        "covered_lines": max(c["covered_lines"] for c in present),
        "num_statements": max(c["num_statements"] for c in present),
        "missing_lines": min(c["missing_lines"] for c in present),
    }


def combine_test_file_stats(
    stats: List[Optional[Dict[str, int]]],
) -> Optional[Dict[str, int]]:
    present = [s for s in stats if s is not None]
    if not present:
        return None
    return {
        "lines": sum(s["lines"] for s in present),
        "test_functions": sum(s["test_functions"] for s in present),
    }


# ---------------------------------------------------------------------------
# Report row building
# ---------------------------------------------------------------------------
def fmt(value) -> str:
    """Render None as empty string, everything else as-is, for CSV output."""
    return "" if value is None else value


def safe_div(numerator, denominator):
    if not denominator:
        return None
    return round(numerator / denominator, 2)


def build_report_rows(
    traditional_run: Optional[Dict[str, object]],
    agentic_run: Optional[Dict[str, object]],
    traditional_cov: Optional[Dict[str, float]],
    agentic_cov: Optional[Dict[str, float]],
    traditional_stats: Optional[Dict[str, int]],
    agentic_stats: Optional[Dict[str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    def add_row(metric, traditional_value, agentic_value, delta=None):
        rows.append(
            {
                "metric": metric,
                "traditional": fmt(traditional_value),
                "agentic": fmt(agentic_value),
                "delta_agentic_minus_traditional": fmt(delta),
            }
        )

    # --- Test execution metrics --------------------------------------
    t_total = traditional_run["total"] if traditional_run else None
    a_total = agentic_run["total"] if agentic_run else None
    add_row(
        "tests_executed",
        t_total,
        a_total,
        (a_total - t_total) if (t_total is not None and a_total is not None) else None,
    )

    t_passed = traditional_run["passed"] if traditional_run else None
    a_passed = agentic_run["passed"] if agentic_run else None
    add_row(
        "tests_passed",
        t_passed,
        a_passed,
        (
            (a_passed - t_passed)
            if (t_passed is not None and a_passed is not None)
            else None
        ),
    )

    t_failed = traditional_run["failed"] if traditional_run else None
    a_failed = agentic_run["failed"] if agentic_run else None
    add_row(
        "tests_failed",
        t_failed,
        a_failed,
        (
            (a_failed - t_failed)
            if (t_failed is not None and a_failed is not None)
            else None
        ),
    )

    t_pass_rate = (
        safe_div(traditional_run["passed"], traditional_run["total"])
        if traditional_run
        else None
    )
    a_pass_rate = (
        safe_div(agentic_run["passed"], agentic_run["total"]) if agentic_run else None
    )
    add_row(
        "pass_rate",
        t_pass_rate,
        a_pass_rate,
        (
            round(a_pass_rate - t_pass_rate, 2)
            if (t_pass_rate is not None and a_pass_rate is not None)
            else None
        ),
    )

    # --- Coverage metrics ----------------------------------------------
    t_cov = traditional_cov["percent_covered"] if traditional_cov else None
    a_cov = agentic_cov["percent_covered"] if agentic_cov else None
    add_row(
        "coverage_percent",
        t_cov,
        a_cov,
        round(a_cov - t_cov, 2) if (t_cov is not None and a_cov is not None) else None,
    )

    t_covered_lines = traditional_cov["covered_lines"] if traditional_cov else None
    a_covered_lines = agentic_cov["covered_lines"] if agentic_cov else None
    add_row(
        "covered_lines",
        t_covered_lines,
        a_covered_lines,
        (
            (a_covered_lines - t_covered_lines)
            if (t_covered_lines is not None and a_covered_lines is not None)
            else None
        ),
    )

    # Coverage retained per test run, i.e. how much coverage "per test"
    # the agentic selection achieves relative to running everything —
    # this is the core "predictive test selection" research metric:
    # did fewer tests still cover most of the code?
    if t_cov and a_cov:
        retention_pct = round((a_cov / t_cov) * 100, 2) if t_cov else None
        add_row("coverage_retained_vs_traditional_pct", "", retention_pct, "")
    else:
        add_row("coverage_retained_vs_traditional_pct", "", None, "")

    # --- Manual-effort / selection-efficiency metrics -------------------
    t_lines = traditional_stats["lines"] if traditional_stats else None
    a_lines = agentic_stats["lines"] if agentic_stats else None
    add_row(
        "test_code_lines",
        t_lines,
        a_lines,
        (a_lines - t_lines) if (t_lines is not None and a_lines is not None) else None,
    )

    t_funcs = traditional_stats["test_functions"] if traditional_stats else None
    a_funcs = agentic_stats["test_functions"] if agentic_stats else None
    add_row(
        "test_functions_authored",
        t_funcs,
        a_funcs,
        (a_funcs - t_funcs) if (t_funcs is not None and a_funcs is not None) else None,
    )

    # Selection ratio: what fraction of all authored tests did the agent
    # actually choose to run? Lower is "more selective" (more effort saved).
    if t_funcs and a_total is not None:
        selection_ratio = safe_div(a_total, t_funcs)
        add_row("tests_run_as_fraction_of_authored", "", selection_ratio, "")
    else:
        add_row("tests_run_as_fraction_of_authored", "", None, "")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare traditional (run-everything) vs agentic (predictively-selected) test results."
    )
    parser.add_argument("--first-test-csv", default="first_test.csv")
    parser.add_argument("--first-test-coverage", default="first_test_coverage.json")
    parser.add_argument("--second-test-csv", default="second_test.csv")
    parser.add_argument("--second-test-coverage", default="second_test_coverage.json")
    parser.add_argument("--selector-csv", default="selecter_results.csv")
    parser.add_argument("--selector-coverage", default="coverage_selector.json")
    parser.add_argument(
        "--traditional-test-files",
        nargs="*",
        default=["tests/test.py", "tests_generated/generated_test.py"],
        help="Test source files whose line/function counts represent the "
        "traditional side's authoring effort.",
    )
    parser.add_argument(
        "--out",
        default="reports/comparison_report.csv",
        help="Where to write the comparison CSV. Parent directory is created if missing.",
    )
    args = parser.parse_args()

    print("📊 Reading traditional pipeline artifacts...")
    first_run = read_pytest_csv(args.first_test_csv)
    second_run = read_pytest_csv(args.second_test_csv)
    if first_run is None:
        print(f"⚠️  Missing or unreadable: {args.first_test_csv}")
    if second_run is None:
        print(f"⚠️  Missing or unreadable: {args.second_test_csv}")
    traditional_run = combine_run_summaries([first_run, second_run])

    first_cov = read_coverage_json(args.first_test_coverage)
    second_cov = read_coverage_json(args.second_test_coverage)
    if first_cov is None:
        print(f"⚠️  Missing or unreadable: {args.first_test_coverage}")
    if second_cov is None:
        print(f"⚠️  Missing or unreadable: {args.second_test_coverage}")
    traditional_cov = combine_coverage([first_cov, second_cov])

    print("📊 Reading agentic pipeline artifacts...")
    agentic_run = read_pytest_csv(args.selector_csv)
    if agentic_run is None:
        print(
            f"⚠️  Missing or unreadable: {args.selector_csv} "
            "(this is expected if the selector ran with zero tests selected)"
        )
    agentic_cov = read_coverage_json(args.selector_coverage)
    if agentic_cov is None:
        print(
            f"⚠️  Missing or unreadable: {args.selector_coverage} "
            "(this is expected if the selector ran with zero tests selected)"
        )

    print("📊 Counting manual-effort proxies (lines, test functions)...")
    traditional_stats = combine_test_file_stats(
        [count_test_file_stats(p) for p in args.traditional_test_files]
    )
    for p in args.traditional_test_files:
        if count_test_file_stats(p) is None:
            print(f"⚠️  Missing or unreadable test source file: {p}")
    # The "agentic" side authors zero new test code of its own — it
    # selects from the same pool the traditional side wrote. Its
    # authoring-effort proxy is therefore 0 lines / 0 functions by
    # definition, not "no data" — this is a deliberate research result
    # (the agent's value is in selection, not authorship).
    agentic_stats = {"lines": 0, "test_functions": 0}

    rows = build_report_rows(
        traditional_run,
        agentic_run,
        traditional_cov,
        agentic_cov,
        traditional_stats,
        agentic_stats,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "metric",
                    "traditional",
                    "agentic",
                    "delta_agentic_minus_traditional",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"✅ Wrote comparison report to {args.out} ({len(rows)} metric rows)")
    except Exception as e:
        print(f"❌ Could not write comparison report to {args.out}: {e}")
        sys.exit(1)

    print("\n📋 Summary:")
    for row in rows:
        print(
            f"   {row['metric']:38s} traditional={row['traditional']!s:10s} agentic={row['agentic']!s:10s} delta={row['delta_agentic_minus_traditional']}"
        )


if __name__ == "__main__":
    main()
