"""Shows the model chain the system uses right now: what Google lists, which models are in the chain, and the health of each
(data/model_health.json). With --probe it also asks every model of the chain one trivial question ("ping"), which spends a real
request of that model's daily quota, and reports the result to the health record (an answer clears the failures in a row; a failure counts;
a quota limit is not a failure; the order of the chain never changes).

Nothing needs to be edited when a model appears or disappears: src/lifecompass/modelchain.py finds the models by asking Google and
keeps them newest first: nothing moves them. To pin models yourself, set GEMINI_MODELS (and GEMINI_MODELS_LISTEN) in .env: a pin wins."""
import json
import os
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from lifecompass import llm, modelchain, tunables

if not os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

pinned = (os.environ.get("GEMINI_MODELS") or "").strip()
print("Pinned in .env:", pinned or "(nothing: the chain is discovered)")
print(f"Stable {tunables.get_text('models', 'family')} models Google lists:", ", ".join(modelchain.candidates()) or "(none: Google could not be asked)")
chain = llm._get_models("extract[1]")
print("Chain used now (extract):", ", ".join(chain))
print("Chain used now (listen): ", ", ".join(llm._get_models("listen[1/1]")))

health = json.loads(modelchain._HEALTH_PATH.read_text(encoding="utf-8")).get("models", {}) if modelchain._HEALTH_PATH.exists() else {}
for name in chain:
    entry = health.get(name) or {}
    print(f"  {name}: failures in a row: {entry.get('failures', 0)}; last failure: {entry.get('last_failure', '-')}; "
          f"last answer: {entry.get('last_ok', '-')}")

if "--probe" in sys.argv:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print("\nProbing the chain with one trivial request each...\n")
    for name in chain:
        try:
            client.models.generate_content(model=name, contents="ping")
            print(f"  OK    {name}")
            modelchain.record(name, "ok")
        except Exception as e:
            print(f"  FAIL  {name}: {str(e)[:120]}")
            if not ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                modelchain.record(name, "fail")
else:
    print("\n(add --probe to test each model of the chain; it spends one request of each model's daily quota)")
