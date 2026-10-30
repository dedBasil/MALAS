import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "gpt-4o-mini"


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call(client: OpenAI, system: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def ask_json(system: str, prompt: str, required: dict) -> dict:
    result = _ask_json(system, prompt, required)
    # Kept for the audit log (audit.py); pipeline.run() strips it from what callers get back.
    result["llm_call"] = {"model": MODEL, "temperature": 0, "system": system, "prompt": prompt}
    return result


def _ask_json(system: str, prompt: str, required: dict) -> dict:
    """Call the LLM and validate the reply. `required` maps key -> allowed values (None = any).

    Never returns a fake verdict: any failure comes back as {"status": "error", "error": ...}.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"status": "error", "error": "OPENAI_API_KEY not set (add it to a .env file)"}
    try:
        result = _call(OpenAI(api_key=api_key), system, prompt)
    except Exception as e:
        return {"status": "error", "error": f"LLM call failed: {e}"}

    for key, allowed in required.items():
        if key not in result:
            return {"status": "error", "error": f"LLM reply missing '{key}'", "raw": result}
        if allowed is not None and result[key] not in allowed:
            return {"status": "error", "error": f"LLM reply has invalid {key}={result[key]!r}", "raw": result}
    result["status"] = "ok"
    return result
