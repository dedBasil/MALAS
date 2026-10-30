"""Append-only audit trail: one JSON line per application, written by pipeline.stream_run().

Look up an application's trail:  python audit.py <application_id | audit_id>
"""
import json
import sys
import threading
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "logs" / "audit.jsonl"
_lock = threading.Lock()  # evaluate.py runs applications in parallel threads


def write(record: dict):
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _lock:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_all() -> list:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find(key: str) -> list:
    return [r for r in find_all() if key in (r["audit_id"], r["application_id"])]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    matches = find(sys.argv[1])
    if not matches:
        sys.exit(f"No audit record for {sys.argv[1]} in {LOG_PATH}")
    for r in matches:
        print(f"=== audit {r['audit_id']}  application {r['application_id']}  {r['started_at']} -> {r['finished_at']}")
        print(f"    final decision {r['final_decision']}   LLM opinion {r['llm_decision']}")
        for step in r["steps"]:
            out = step["output"]
            llm = out.get("llm_call")
            print(f"  [{step['finished_at']}] {step['agent']:<12} +{step['elapsed_ms']:>6} ms  "
                  f"status={out.get('status', 'ok')}" + (f"  llm={llm['model']}" if llm else ""))
        print(json.dumps(r, indent=2, ensure_ascii=False))
