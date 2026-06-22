import asyncio
import json
import os
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
- Uses the `client` fixture (httpx.AsyncClient).
- Covers the happy path and the main error cases (404, 422 validation errors).
- For GET: test a valid ID and an invalid/non‑existent ID.
- For POST: test valid creation, invalid payload (blank/too long), and user not found (if applicable).
- For PUT: test update, 404, validation errors.
- For DELETE: test deletion, then 404 on second attempt.
- Asserts both status codes and response structure (check presence of keys like "id", "user", etc.).

Name the function: `test_{method}_api_{path.replace("/", "_").strip("_")}`

Return **only the Python code**, no markdown fences, no extra text.
"""
    return prompt


# ---------------------------------------------------------------------------
# Step 3 – Call OpenRouter (non‑streaming)
# ---------------------------------------------------------------------------
async def generate_test(prompt):
    try:
        # Offload the synchronous OpenAI/OpenRouter client call to an async thread pool
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            extra_headers={
                "HTTP-Referer": "https://localhost:8000",  # Optional OpenRouter leaderboard tracking
                "X-Title": "FastAPI Test Generator",
            },
        )

        # OpenRouter extracts text structure identically to standard OpenAI completions
        code = response.choices[0].message.content.strip()

        # Remove markdown code fences if present
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

    except Exception as e:
        print(f"⚠️  OpenRouter Gemini call failed: {e}")
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

    # Save the file using the absolute output path
    with open(OUTPUT_PATH, encoding="utf-8", mode="w") as f:
        f.write("import pytest\nimport httpx\nfrom main import app\n\n")
        f.write(
            "# ---------------------------------------------------------------------------\n"
        )
        f.write("# Generated tests – review before committing\n")
        f.write(
            "# ---------------------------------------------------------------------------\n\n"
        )
        f.write("\n\n".join(all_tests))

    print(f"✅ Wrote {len(all_tests)} tests to {OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Step 5 – Run pytest with coverage
    # -----------------------------------------------------------------------
    print("\n🧪 Running generated tests with coverage...\n")
    result = subprocess.run(
        [
            "pytest",
            OUTPUT_PATH,  # Use the dynamic path here too!
            "--cov=main",
            "--cov-report=term-missing",
            "--tb=short",
        ],
        capture_output=False,
        text=True,
    )


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
