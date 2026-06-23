import ast
import asyncio
import json
import os
import re
import subprocess
import sys

import httpx
from dotenv import load_dotenv
from openai import OpenAI  # Use OpenRouter via OpenAI-compatible SDK

# ---------------------------------------------------------------------------
# Load .env and set OpenRouter client
# ---------------------------------------------------------------------------
load_dotenv()

# OpenRouter utilizes its own API key structure
if not os.getenv("OPENROUTER_API_KEY"):
    print(
        "❌ ERROR: No OpenRouter API key found in .env (expected 'OPENROUTER_API_KEY')"
    )
    sys.exit(1)

# Use OpenRouter's identifier for Gemini models
# Valid options include: "google/gemini-2.5-flash" or "google/gemini-2.5-pro"
MODEL_NAME = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")
print(f"🧠 Using OpenRouter Gemini model: {MODEL_NAME}")

# Initialize OpenAI client pointed directly to OpenRouter's endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

SERVER_URL = "http://localhost:8000"

# Max tokens per generation request. OpenRouter free/low-credit accounts cap
# total tokens-affordable-per-request (your account is currently capped
# around ~1100). Keep this comfortably under that ceiling — override via
# the MAX_TOKENS env var once you add credits and want richer test bodies.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))


# ---------------------------------------------------------------------------
# Step 1 – Fetch OpenAPI spec
# ---------------------------------------------------------------------------
async def fetch_openapi():
    async with httpx.AsyncClient() as http_client:
        resp = await http_client.get(f"{SERVER_URL}/openapi.json")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Step 2 – Build LLM prompt
# ---------------------------------------------------------------------------
def build_prompt(path, method, operation):
    params = operation.get("parameters", [])
    req_body = operation.get("requestBody", {})
    responses = operation.get("responses", {})
    summary = operation.get("summary", "")
    description = operation.get("description", "")

    prompt = f"""You are an expert QA engineer writing pytest-asyncio tests for a FastAPI backend.

Endpoint:
- Path: {path}
- Method: {method.upper()}
- Summary: {summary}
- Description: {description}

Parameters:
{json.dumps(params, indent=2)}

Request Body (if any):
{json.dumps(req_body, indent=2)}

Expected Responses:
{json.dumps(responses, indent=2)}

Write a **single** pytest-asyncio test function that:
- Takes `client` as its only parameter — this is an async httpx.AsyncClient fixture provided by the project's conftest.py. Do not define or import it yourself.
- Covers the happy path and the main error cases (404, 422 validation errors) ONLY if they are actually applicable to this specific endpoint based on the spec above. Skip inapplicable cases silently — do not write comments explaining why a case doesn't apply.
- For GET: test a valid ID and an invalid/non‑existent ID, if the endpoint takes an ID.
- For POST: test valid creation, invalid payload (blank/too long), and user not found, if applicable.
- For PUT: test update, 404, validation errors, if applicable.
- For DELETE: test deletion, then 404 on second attempt, if applicable.
- Asserts both status codes and response structure (check presence of keys like "id", "user", etc.).
- Contains NO comments and NO explanatory prose — only executable code. Every line must be a statement, not a note.

Name the function: `test_{method}_api_{path.replace("/", "_").strip("_")}`

Return **only the Python code**, no markdown fences, no extra text.
"""
    return prompt


# ---------------------------------------------------------------------------
# Step 3a – Extract code from a model response and validate it
# ---------------------------------------------------------------------------
def _extract_code(text: str) -> str:
    """Pull code out of a ```python ... ``` / ``` ... ``` fence if present.

    Using a regex here (rather than fixed-width slicing on startswith) is
    robust to variations like ```py, ```Python, a fence with no trailing
    newline, or stray prose the model added despite being told not to.
    """
    text = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No fences found — assume the whole response is already raw code.
    return text


def _is_valid_python(code: str) -> bool:
    """Quick syntax check so we never write unparseable code to disk."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# Step 3 – Call OpenRouter (non‑streaming), with validation + one retry
# ---------------------------------------------------------------------------
async def generate_test(prompt, retries: int = 1):
    last_invalid_code = None

    for attempt in range(retries + 1):
        try:
            # Offload the synchronous OpenAI/OpenRouter client call to an async thread pool
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                # 800 tokens is too tight for a function covering happy-path +
                # 404 + 422 + structure assertions — truncation mid-function
                # is the most common cause of "syntax/indentation errors".
                # Capped via MAX_TOKENS to also respect OpenRouter credit limits.
                max_tokens=MAX_TOKENS,
                extra_headers={
                    "HTTP-Referer": "https://localhost:8000",  # Optional OpenRouter leaderboard tracking
                    "X-Title": "FastAPI Test Generator",
                },
            )

            raw = response.choices[0].message.content.strip()
            code = _extract_code(raw)

            if _is_valid_python(code):
                return code

            last_invalid_code = code
            remaining = retries - attempt
            print(
                f"   ⚠️  Attempt {attempt + 1} produced invalid Python "
                f"(syntax error) — {'retrying' if remaining > 0 else 'giving up'}."
            )

        except Exception as e:
            err_str = str(e)
            if "402" in err_str or "more credits" in err_str.lower():
                print(
                    "❌ OpenRouter rejected the request for insufficient "
                    "credits/budget at the current max_tokens "
                    f"({MAX_TOKENS}). Lower MAX_TOKENS, or add credits at "
                    "https://openrouter.ai/settings/credits."
                )
            else:
                print(f"⚠️  OpenRouter Gemini call failed: {e}")
            return None

    # All retries exhausted and still invalid — skip rather than corrupt the file.
    if last_invalid_code is not None:
        print("   ⚠️  Skipping this test: could not get valid Python after retries.")
    return None


# ---------------------------------------------------------------------------
# Step 4 – Write generated tests to file and run them
# ---------------------------------------------------------------------------
async def main():
    print("📡 Fetching OpenAPI spec...")
    spec = await fetch_openapi()
    print(f"✅ Found {len(spec['paths'])} endpoints.")

    all_tests = []
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            if method.lower() not in ["get", "post", "put", "delete"]:
                continue
            print(f"🧠 Generating test for {method.upper()} {path} ...")
            prompt = build_prompt(path, method, operation)
            code = await generate_test(prompt)
            if code:
                all_tests.append(code)
            else:
                print(f"⚠️  Skipping {method.upper()} {path} due to generation error.")

    if not all_tests:
        print("❌ No tests were generated. Exiting.")
        return

    # Dynamically locate the script's directory (tests_generated/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "generated_tests.py")

    header = (
        "import pytest\n\n"
        "# ---------------------------------------------------------------------------\n"
        "# Generated tests – review before committing\n"
        "# The `client` and `reset_db` fixtures come from the project-root\n"
        "# conftest.py (auto-discovered by pytest) — no import needed here.\n"
        "# ---------------------------------------------------------------------------\n\n"
    )
    full_file_contents = header + "\n\n".join(all_tests)

    # Defense in depth: each function was validated individually, but verify
    # the fully concatenated file too, in case of duplicate function names or
    # other interaction effects between generations.
    if not _is_valid_python(full_file_contents):
        print(
            "❌ The concatenated test file failed to parse even though each "
            "function validated individually. Writing it anyway for "
            "inspection, but NOT running pytest on it."
        )
        with open(OUTPUT_PATH, encoding="utf-8", mode="w") as f:
            f.write(full_file_contents)
        print(f"📄 Wrote (invalid) file to {OUTPUT_PATH} for manual review.")
        return

    # Save the file using the absolute output path
    with open(OUTPUT_PATH, encoding="utf-8", mode="w") as f:
        f.write(full_file_contents)

    print(f"✅ Wrote {len(all_tests)} tests to {OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Step 5 – Run pytest with coverage, and emit a CSV report
    # -----------------------------------------------------------------------
    print("\n🧪 Running generated tests with coverage...\n")
    REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    CSV_PATH = os.path.join(REPORTS_DIR, "generated_test_results.csv")

    result = subprocess.run(
        [
            "pytest",
            OUTPUT_PATH,  # Use the dynamic path here too!
            "--cov=main",
            "--cov-report=term-missing",
            "--tb=short",
            f"--csv={CSV_PATH}",
            "--csv-columns=id,status,duration,message",
        ],
        capture_output=False,
        text=True,
    )
    print(f"📄 Wrote CSV report to {CSV_PATH}")


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
