import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Load environment and OpenRouter client
# ---------------------------------------------------------------------------
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("❌ ERROR: OpenRouter API key missing (expected 'OPENROUTER_API_KEY')")
    sys.exit(1)

MODEL_NAME = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")
print(f"🧠 Using OpenRouter Gemini model: {MODEL_NAME}")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Always collected by default, regardless of --extra-tests.
DEFAULT_TEST_PATHS = [
    "tests/test.py",
    "tests_generated/generated_test.py",
]


# ---------------------------------------------------------------------------
# Step 1 – Get code diff
# ---------------------------------------------------------------------------
def get_git_diff(commit: str = "HEAD") -> str:
    """Run git diff and return the unified diff as string."""
    try:
        result = subprocess.run(
            ["git", "diff", commit],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print("⚠️  Could not run git diff. Ensure you are in a git repo.")
        return ""


def read_diff_file(path: str) -> str:
    """Read a unified diff from a file on disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  Could not read diff file {path}: {e}")
        return ""


def parse_diff(diff_text: str) -> Dict[str, List[int]]:
    """
    Extract changed files and line numbers from a unified diff.
    Returns a dict: {filename: [line_numbers_of_changes]}.
    """
    changes: Dict[str, List[int]] = {}
    current_file = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # Extract file path
            match = re.search(r" b/(.+)$", line)
            if match:
                current_file = match.group(1)
                changes[current_file] = []
        elif line.startswith("@@") and current_file:
            # Parse hunk header: @@ -a,b +c,d @@
            match = re.search(r"\+(\d+),?(\d+)?", line)
            if match:
                start = int(match.group(1))
                length = int(match.group(2)) if match.group(2) else 1
                # Store the start line and mark changed lines from start to start+length-1
                changes[current_file].extend(range(start, start + length))
    return changes


# ---------------------------------------------------------------------------
# Step 2 – Collect existing tests
#
# Always collects the two default project test files. If --extra-tests was
# passed on the command line, that file or directory is collected in
# addition to (not instead of) the defaults — so by default you get the
# standard 2-file pipeline, and opting in to a path like
# "tests_generated/generated_tests.py" gives you those plus that one.
# ---------------------------------------------------------------------------
def collect_tests(extra_path: Optional[str] = None) -> List[str]:
    paths = list(DEFAULT_TEST_PATHS)

    if extra_path:
        if os.path.exists(extra_path):
            paths.append(extra_path)
        else:
            print(
                f"⚠️  --extra-tests path '{extra_path}' does not exist — "
                f"running the default {len(DEFAULT_TEST_PATHS)} test file(s) only."
            )

    # Skip paths that don't exist so pytest isn't asked to collect a
    # nonexistent default file (e.g. before the agent has ever run, there is
    # no tests_generated/generated_test.py yet).
    existing_paths = [p for p in paths if os.path.exists(p)]
    missing_paths = [p for p in paths if p not in existing_paths]
    for missing in missing_paths:
        print(f"⚠️  Test path '{missing}' does not exist — skipping.")

    if not existing_paths:
        return []

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *existing_paths,
                "--collect-only",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",  # Prevents UnicodeDecodeError on non-ASCII output
        )
        tests = []
        for line in result.stdout.splitlines():
            if "::" in line and "[" not in line:
                tests.append(line.strip())
        if not tests and (
            "error" in result.stdout.lower() or result.returncode not in (0, 5)
        ):
            print("⚠️  pytest reported errors during collection:")
            print(result.stdout[-2000:])
        return tests
    except Exception as e:
        print(f"Failed to collect tests: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 3 – Build LLM prompt for test selection
# ---------------------------------------------------------------------------
def build_selection_prompt(diff_text: str, test_list: List[str]) -> str:
    # Truncate diff if too long
    diff_preview = diff_text[:4000] + ("..." if len(diff_text) > 4000 else "")

    tests_preview = "\n".join(test_list[:50])  # limit to 50 tests for prompt size

    prompt = f"""You are a test-selection assistant for a FastAPI project.

The code diff below shows what changed:
```diff
{diff_preview}
```

Here is the list of all test functions (including file names):
```
{tests_preview}
```

Your task:
1. Analyse the diff and determine which tests are most likely to be affected.
2. Rank them by relevance: high, medium, low.
3. Return a JSON object exactly like this:
{{
  "high": ["test_file::test_func", "..."],
  "medium": ["test_file::test_func", "..."],
  "low": ["test_file::test_func", "..."]
}}

Only include test IDs that exist in the list above. Do not invent names.
If no tests are relevant, return empty lists.
Return only the JSON object, no additional text or markdown.
"""
    return prompt


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json, ```python, ```) from a string."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line (e.g. ``` or ```json or ```python)
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def select_tests(diff_text: str, test_list: List[str]) -> Dict[str, List[str]]:
    prompt = build_selection_prompt(diff_text, test_list)
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            extra_headers={
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "TestSelectAgent",
            },
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_code_fences(raw)
        data = json.loads(raw)

        # Ensure keys exist
        return {
            "high": data.get("high", []),
            "medium": data.get("medium", []),
            "low": data.get("low", []),
        }
    except Exception as e:
        print(f"⚠️  Selection LLM call failed: {e}")
        return {"high": [], "medium": [], "low": []}


# ---------------------------------------------------------------------------
# Step 4 – Self-healing: suggest fix for a failed test
# ---------------------------------------------------------------------------
async def suggest_fix(test_id: str, error_output: str) -> Optional[str]:
    """Ask LLM to fix the test function based on error."""
    test_file = test_id.split("::")[0]
    test_func = test_id.split("::")[1] if "::" in test_id else None

    if not os.path.exists(test_file):
        print(f"⚠️  Test file {test_file} not found.")
        return None

    # Read the test file content
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Could not read {test_file}: {e}")
        return None

    # We'll send the whole file (truncated) rather than try to isolate the function
    content_preview = content[:4000] + ("…" if len(content) > 4000 else "")

    prompt = f"""The test {test_id} failed with this error:
```
{error_output[:2000]}
```

Here is the content of the test file {test_file}:
```python
{content_preview}
```

Please suggest a corrected version of the test function {test_func if test_func else "the failing test"}.
The failure may be due to an API change (e.g., renamed field, changed status code, or updated validation).

Return only the corrected function code (the whole function), no markdown fences.
"""

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
            extra_headers={
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "TestSelectAgent (heal)",
            },
        )
        fixed = response.choices[0].message.content.strip()
        fixed = _strip_code_fences(fixed)
        return fixed
    except Exception as e:
        print(f"⚠️  Self-heal LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CSV reporting helper
# ---------------------------------------------------------------------------
def _write_csv(path: Optional[str], rows: List[Dict[str, str]]) -> None:
    """Write the selected/executed tests to a CSV with columns:
    id,status,duration,message

    Writes a header-only CSV when there are no rows, so downstream steps
    (e.g. actions/upload-artifact) always have a file to pick up, and so the
    artifact is consistent in shape with the main pytest --csv report.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "status", "duration", "message"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"📄 Wrote CSV report to {path} ({len(rows)} row(s))")
    except Exception as e:
        print(f"⚠️  Could not write CSV report to {path}: {e}")


# ---------------------------------------------------------------------------
# Step 5 – Main driver
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="TestSelectAgent")
    parser.add_argument("--diff", help="Path to a diff file (instead of git diff)")
    parser.add_argument(
        "--run-medium", action="store_true", help="Also run medium-relevance tests"
    )
    parser.add_argument("--commit", default="HEAD", help="Git commit to diff against")
    parser.add_argument(
        "--no-heal", action="store_true", help="Skip self-healing attempts"
    )
    parser.add_argument(
        "--extra-tests",
        default=None,
        help="Additional test file or directory to collect ON TOP OF the "
        f"default {len(DEFAULT_TEST_PATHS)} files ({', '.join(DEFAULT_TEST_PATHS)}). "
        "E.g. --extra-tests tests_generated/generated_tests.py to also "
        "include the latest TestGenAgent output.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Path to write a CSV report (columns: id,status,duration,message) "
        "for the tests that were actually selected and run",
    )
    args = parser.parse_args()

    # Obtain diff
    if args.diff:
        diff_text = read_diff_file(args.diff)
    else:
        diff_text = get_git_diff(commit=args.commit)

    if not diff_text:
        print("❌ No changes detected. Exiting.")
        _write_csv(args.csv_out, [])
        return

    print("📝 Analysing code changes…")
    changed_files = parse_diff(diff_text)
    if changed_files:
        print("📄 Changed files:")
        for f, lines in changed_files.items():
            preview = lines[:5]
            suffix = "…" if len(lines) > 5 else ""
            print(f"   {f} (lines {preview}{suffix})")
    else:
        print("⚠️  Could not parse diff line numbers (will still use full diff).")

    # Collect tests
    if args.extra_tests:
        print(
            f"📋 Collecting default tests ({', '.join(DEFAULT_TEST_PATHS)}) "
            f"plus extra path '{args.extra_tests}'…"
        )
    else:
        print(f"📋 Collecting default tests ({', '.join(DEFAULT_TEST_PATHS)})…")
    tests = collect_tests(extra_path=args.extra_tests)
    if not tests:
        print("❌ No tests found. Are you in the correct directory?")
        _write_csv(args.csv_out, [])
        return
    print(f"📋 Found {len(tests)} tests.")

    # Select tests
    print("🧠 Asking LLM to select relevant tests…")
    selection = await select_tests(diff_text, tests)
    high = selection.get("high", [])
    medium = selection.get("medium", [])
    low = selection.get("low", [])

    print("\n🎯 Selected tests:")
    print(f"   HIGH ({len(high)}):")
    for t in high:
        print(f"     - {t}")
    print(f"   MEDIUM ({len(medium)}):")
    for t in medium:
        print(f"     - {t}")
    print(f"   LOW ({len(low)}):")
    for t in low:
        print(f"     - {t}")

    # Decide which to run
    to_run = high[:]
    if args.run_medium:
        to_run.extend(medium)
    if not to_run:
        print("✅ No tests selected for execution. Exiting.")
        _write_csv(args.csv_out, [])
        return

    print(f"\n🧪 Running {len(to_run)} tests…")
    csv_rows: List[Dict[str, str]] = []
    coverage_data_file = os.path.join(
        os.path.dirname(args.csv_out) if args.csv_out else ".", ".coverage.selector"
    )
    if os.path.exists(coverage_data_file):
        os.remove(coverage_data_file)

    for test_id in to_run:
        print(f"\n▶️  {test_id}")
        start = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                test_id,
                "-v",
                "--tb=short",
                "--no-header",
                "--cov=main",
                "--cov-append",
                "--cov-report=",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "COVERAGE_FILE": coverage_data_file},
        )
        duration = round(time.monotonic() - start, 3)

        if result.returncode == 0:
            print("   ✅ PASSED")
            csv_rows.append(
                {
                    "id": test_id,
                    "status": "passed",
                    "duration": duration,
                    "message": "",
                }
            )
        else:
            print("   ❌ FAILED")
            failure_message = (result.stdout + "\n" + result.stderr).strip()
            if not args.no_heal:
                print("   🔧 Attempting self-heal…")
                fixed = await suggest_fix(test_id, failure_message)
                if fixed:
                    print("   💡 Suggested fix (review and apply):")
                    print("   " + "\n   ".join(fixed.splitlines()))
                    print(
                        f"   ℹ️  Replace the old function in {test_id.split('::')[0]}"
                    )
                    failure_message = (
                        f"{failure_message[:500]} | self-heal suggestion generated"
                    )
                else:
                    print("   ⚠️  Could not auto-heal. Please fix manually.")
            else:
                print("   ℹ️  Self-heal disabled by --no-heal.")

            # Keep the CSV message field short and single-line for readability.
            short_message = " ".join(failure_message.split())[:300]
            csv_rows.append(
                {
                    "id": test_id,
                    "status": "failed",
                    "duration": duration,
                    "message": short_message,
                }
            )

    _write_csv(args.csv_out, csv_rows)

    coverage_json_path = os.path.join(
        os.path.dirname(args.csv_out) if args.csv_out else ".", "coverage_selector.json"
    )
    if os.path.exists(coverage_data_file):
        subprocess.run(
            ["coverage", "json", "-o", coverage_json_path],
            env={**os.environ, "COVERAGE_FILE": coverage_data_file},
            capture_output=True,
            text=True,
        )
        print(f"📄 Wrote coverage report to {coverage_json_path}")
    else:
        print("ℹ️  No coverage data file produced (no tests ran with coverage).")
    print("\n✅ Test selection and execution complete.")


if __name__ == "__main__":
    asyncio.run(main())
