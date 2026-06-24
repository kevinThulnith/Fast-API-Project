import csv
import os
import sys
from typing import Dict, List

# ---------------------------------------------------------------------------
# Comparison Report Generator
#
# Compares the "traditional" run (full suite, every test executed — i.e.
# test-results.csv from `pytest tests/ tests_generated/`) against the
# "agentic" run (LLM-selected subset — reports/test_results.csv from
# selecter.py) and writes a single comparison CSV summarizing the delta.
# ---------------------------------------------------------------------------


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path} (treating as empty)")
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    total = len(rows)
    passed = sum(1 for r in rows if r.get("status") == "passed")
    failed = sum(1 for r in rows if r.get("status") == "failed")
    duration = sum(float(r.get("duration") or 0) for r in rows)
    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "total_duration_sec": round(duration, 3),
    }


def build_comparison_rows(
    traditional: Dict[str, float], agentic: Dict[str, float]
) -> List[Dict[str, str]]:
    def pct_change(old: float, new: float) -> str:
        if old == 0:
            return "n/a"
        return f"{round(((new - old) / old) * 100, 1)}%"

    metrics = [
        ("total_tests", "Total tests executed"),
        ("passed", "Tests passed"),
        ("failed", "Tests failed"),
        ("total_duration_sec", "Total execution time (s)"),
    ]

    rows = []
    for key, label in metrics:
        t_val = traditional.get(key, 0)
        a_val = agentic.get(key, 0)
        rows.append(
            {
                "metric": label,
                "traditional": t_val,
                "agentic": a_val,
                "delta": round(a_val - t_val, 3),
                "pct_change": pct_change(t_val, a_val),
            }
        )

    # Derived efficiency metric: % of tests skipped by predictive selection
    skip_pct = "n/a"
    if traditional.get("total_tests", 0) > 0:
        skipped = traditional["total_tests"] - agentic.get("total_tests", 0)
        skip_pct = f"{round((skipped / traditional['total_tests']) * 100, 1)}%"
    rows.append(
        {
            "metric": "Tests skipped by predictive selection",
            "traditional": "-",
            "agentic": "-",
            "delta": "-",
            "pct_change": skip_pct,
        }
    )

    return rows


def write_comparison_csv(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "traditional", "agentic", "delta", "pct_change"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"📄 Wrote comparison report to {path}")


def main():
    traditional_path = sys.argv[1] if len(sys.argv) > 1 else "test-results.csv"
    agentic_path = sys.argv[2] if len(sys.argv) > 2 else "reports/test_results.csv"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "reports/comparison_report.csv"

    print(f"📊 Traditional run (full suite): {traditional_path}")
    print(f"📊 Agentic run (predictive selection): {agentic_path}")

    traditional_rows = read_csv_rows(traditional_path)
    agentic_rows = read_csv_rows(agentic_path)

    traditional_summary = summarize(traditional_rows)
    agentic_summary = summarize(agentic_rows)

    print(f"   Traditional: {traditional_summary}")
    print(f"   Agentic:     {agentic_summary}")

    comparison_rows = build_comparison_rows(traditional_summary, agentic_summary)
    write_comparison_csv(output_path, comparison_rows)


if __name__ == "__main__":
    main()
