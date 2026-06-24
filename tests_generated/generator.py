import ast
import asyncio
import json
import os
import re
import subprocess
import sys
from typing import Optional

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
# Step 1b – Build a plausible request body from a JSON Schema (best-effort)
# ---------------------------------------------------------------------------
def _resolve_schema(schema: dict, components: dict) -> dict:
    """Resolve a single level of $ref against components.schemas."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        return components.get("schemas", {}).get(ref_name, {})
    return schema


def _sample_value_for_property(name: str, prop: dict):
    """Generate a small, schema-valid placeholder value for one field."""
    p_type = prop.get("type")
    if "minimum" in prop:
        return prop["minimum"]
    if p_type == "integer":
        return 1
    if p_type == "number":
        return 1.0
    if p_type == "boolean":
        return True
    if p_type == "array":
        return []
    if p_type == "string":
        fmt = prop.get("format")
        if fmt == "email" or "email" in name.lower():
            return "sample@example.com"
        min_len = prop.get("minLength", 1)
        return ("sample text " * 3)[: max(min_len, len("sample"))] or "sample"
    return "sample"


def build_sample_body(operation: dict, spec: dict) -> Optional[dict]:
    """Build a minimal valid JSON body from requestBody schema, if any."""
    req_body = operation.get("requestBody", {})
    if not req_body:
        return None
    content = req_body.get("content", {}).get("application/json", {})
    schema = content.get("schema", {})
    schema = _resolve_schema(schema, spec.get("components", {}))
    properties = schema.get("properties", {})
    if not properties:
        return None
    return {
        name: _sample_value_for_property(name, prop)
        for name, prop in properties.items()
    }


def build_sample_path(path: str) -> str:
    """Replace {param} placeholders in a path with a safe sample value (1)."""
    return re.sub(r"\{[^}]+\}", "1", path)


# ---------------------------------------------------------------------------
# Step 1c – Call the live server once per endpoint to get a REAL response.
#
# This is the fix for hallucinated response shapes: instead of asking the
# LLM to guess field names from a possibly-empty OpenAPI response schema,
# we show it one real, ground-truth JSON response from the running server.
# Best-effort only — if the call fails (e.g. needs a real existing ID),
# we fall back to "no sample available" and the LLM uses the spec alone.
# ---------------------------------------------------------------------------
async def fetch_sample_response(
    http_client: httpx.AsyncClient, path: str, method: str, operation: dict, spec: dict
) -> Optional[dict]:
    sample_path = build_sample_path(path)
    url = f"{SERVER_URL}{sample_path}"
    try:
        if method.lower() == "get":
            resp = await http_client.get(url, timeout=5.0)
        elif method.lower() == "delete":
            # Never actually delete data while sampling — skip live-fetch for DELETE.
            return None
        elif method.lower() in ("post", "put"):
            body = build_sample_body(operation, spec)
            # Don't let a sampling call mutate state in ways that break later
            # tests (e.g. don't POST /api/posts for real). Only sample safe,
            # idempotent-ish endpoints; skip mutating writes.
            if path.rstrip("/") in ("/api/reset", "/api/seed", "/api/error-injection"):
                return None
            resp = await http_client.request(
                method.upper(), url, json=body, timeout=5.0
            )
        else:
            return None

        if resp.status_code >= 500:
            return None
        try:
            return {"status_code": resp.status_code, "json": resp.json()}
        except ValueError:
            return {"status_code": resp.status_code, "json": None}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 2 – Build LLM prompt
# ---------------------------------------------------------------------------
def build_prompt(path, method, operation, sample_response: Optional[dict] = None):
    params = operation.get("parameters", [])
    req_body = operation.get("requestBody", {})
    responses = operation.get("responses", {})
    summary = operation.get("summary", "")
    description = operation.get("description", "")

    if sample_response is not None:
        sample_block = f"""
Real Live Response (ground truth — fetched from the running server just now,
status {sample_response["status_code"]}):
{json.dumps(sample_response["json"], indent=2)}

This is the ACTUAL response shape. Base your assertions on these exact keys
and structure — do NOT assume different field names than what is shown here.
"""
    else:
        sample_block = """
No live sample response was available for this endpoint (e.g. it mutates
state or requires an existing resource). Rely on the OpenAPI schema below,
and keep structural assertions conservative (e.g. assert response.json() is
a dict/list) rather than asserting specific field names you are not sure of.
"""

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

Expected Responses (OpenAPI spec):
{json.dumps(responses, indent=2)}
{sample_block}
CRITICAL — exact path string: use the path EXACTLY as
"{path}" (with parameter values substituted) for every request you make to
this endpoint, including in any setup/arrange step. Do NOT add or remove a
trailing slash — "{path}" and "{path}/" are different routes and the wrong
one returns 307, not the response you expect.

CRITICAL — numeric validation boundaries: when the Request Body schema above
specifies "minimum"/"maximum" (or ge/le) on a numeric field, an "invalid"
test value must be strictly outside that range (e.g. minimum - 1 or
maximum + 1), not just any number you guess. Re-read the schema's
minimum/maximum values above before writing the invalid-payload test case.

Write a **single** pytest-asyncio test function that:
- Takes `client` as its only parameter — this is an async httpx.AsyncClient fixture provided by the project's conftest.py. Do not define or import it yourself.
- Covers the happy path and the main error cases (404, 422 validation errors) ONLY if they are actually applicable to this specific endpoint based on the spec above. Skip inapplicable cases silently — do not write comments explaining why a case doesn't apply.
- For GET: test a valid ID and an invalid/non‑existent ID, if the endpoint takes an ID.
- For POST: test valid creation, invalid payload (blank/too long/out-of-range per the boundary rule above), and user not found, if applicable.
- For PUT: test update, 404, validation errors, if applicable.
- For DELETE: test deletion, then 404 on second attempt, if applicable.
- Asserts status codes, and asserts response structure using ONLY the keys shown in the Real Live Response above (if provided) — do not invent field names.
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
    async with httpx.AsyncClient() as http_client:
        for path, methods in spec["paths"].items():
            for method, operation in methods.items():
                if method.lower() not in ["get", "post", "put", "delete"]:
                    continue
                print(f"🧠 Generating test for {method.upper()} {path} ...")
                sample_response = await fetch_sample_response(
                    http_client, path, method, operation, spec
                )
                if sample_response is not None:
                    print(
                        f"   📦 Got live sample response (status {sample_response['status_code']})"
                    )
                else:
                    print("   ⚠️  No live sample available — falling back to spec only")
                prompt = build_prompt(path, method, operation, sample_response)
                code = await generate_test(prompt)
                if code:
                    all_tests.append(code)
                else:
                    print(
                        f"⚠️  Skipping {method.upper()} {path} due to generation error."
                    )

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
