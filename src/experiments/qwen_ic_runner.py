"""IC component on a Qwen (OpenAI-compatible) endpoint, for the Wu anchor leg and
the full SPA leg. Faithful to the released protocol: one call per participant,
the participant's own training split as history, testing split as queries, the
ORIGINAL confidence elicitation (v2's exception-aware riders removed).

Env (from ../.env.qwen): OPENAI_API_KEY, OPENAI_BASE_URL, QWEN_MODEL.
Usage: python3 qwen_ic_runner.py wu|spa [--limit N] [--workers 8]
Outputs: results{,_spa}/qwen_ic_predictions_<corpus>.json (+ raw/ per-participant cache)
"""
import json, os, re, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for line in open(os.path.join(ROOT, ".env.qwen")):
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k, v)
API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ["OPENAI_BASE_URL"].rstrip("/")
MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

EXAMPLE_OUTPUTS = '''
[
    {
        "query": "I want to lose weight. Can you create an at-home full-body workout plan for me?",
        "id": 0,
        "permission": {
            "Fitness Tracking,Fitness goal (i.e. weight loss)": {"label": "Yes, always share", "score": 0.92},
            "Fitness Tracking,Gender": {"label": "Yes, always share", "score": 0.88}
        }
    }
]
'''

def make_prompt(bio_info, ai_experiences, permission_history, query_list):
    # Original elicitation: v2's exception-aware sentences removed, all else identical.
    return (
        "You are an intelligent, personalized assistant tasked with predicting a user's permission decisions within an LLM-based agentic system. "
        "Your task is to analyze the following data and output, for each permission request, a prediction label along with a confidence score between 0 and 1. "
        "The factors to consider are:\n"
        "- User demographic and bio information\n"
        "- AI tool usage experiences\n"
        "- Permission decision history\n"
        "- The semantic meaning of the new permission request\n"
        "The output should include two keys for each permission request:\n"
        "1. 'label': with either 'Yes, always share' or 'No, never share'\n"
        "2. 'score': the confidence score of the prediction\n\n"
        "Please follow the JSON format exactly, without any extra text. Use the following format as an example:\n"
        f"{EXAMPLE_OUTPUTS}\n\n"
        "User's demographic and bio information:\n"
        f"{bio_info}\n\n"
        "User's AI tool usage experiences:\n"
        f"{ai_experiences}\n\n"
        "User's permission decision history:\n"
        f"{permission_history}\n\n"
        "New permission requests:\n"
        f"{query_list}\n\n"
        "For each permission request, output the decision (label) and the corresponding confidence score.\n"
        "IMPORTANT1: Your output must be in valid JSON format exactly, without any extra text or commentary outside the JSON structure.\n"
        "IMPORTANT2: All numeric fields (such as the score) must be valid numbers (e.g. 0.90) and not words.\n"
        "IMPORTANT3: Neither the label nor the score fields should be None."
    )

def call_llm(prompt, max_retries=5):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                BASE_URL + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.load(r)
            return out["choices"][0]["message"]["content"], out.get("usage", {})
        except Exception as e:
            wait = min(4 * 2 ** attempt, 60)
            sys.stderr.write(f"  retry {attempt+1} after {wait}s: {e}\n")
            time.sleep(wait)
    raise RuntimeError("exhausted retries")

def extract_predictions(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        return json.loads(m.group(0) if m else text)
    except Exception:
        return []

BIO_CODE = {"age": "age-group code", "gender": "gender code", "education": "education-level code"}
def spa_bio(p):
    return "; ".join(f"{BIO_CODE[k]}: {p.get(k)}" for k in ("age", "gender", "education"))
def spa_aiexp(p):
    return ("AI familiarity (1-5): {ai_familiarity}; AI usage frequency (1-5): {ai_frequency}; "
            "AI trust (1-5): {ai_trust}; privacy importance (1-5): {privacy_importance}"
            ).format(**{k: p.get(k) for k in ("ai_familiarity", "ai_frequency", "ai_trust", "privacy_importance")})

def process(pid, p, corpus, rawdir):
    cache = os.path.join(rawdir, f"{pid}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    if corpus == "wu":
        bio, aiexp = p["bio"], p["ai_experience"]
        queries = [{"query": ex.get("query", ""), "id": ex["id"],
                    "permission": {k: None for k in ex.get("answer", ex.get("permission", {}))}}
                   for ex in p.get("testing", [])]
    else:
        bio, aiexp = spa_bio(p), spa_aiexp(p)
        queries = [{"id": ex["id"],
                    "permission": {k: None for k in ex.get("answer", {})}}
                   for ex in p.get("testing", [])]
    if not queries:
        rec = {"participant_id": pid, "predictions": [], "skipped": True}
        json.dump(rec, open(cache, "w")); return rec
    prompt = make_prompt(bio, aiexp, p.get("training", []), queries)
    text, usage = call_llm(prompt)
    rec = {"participant_id": pid, "model": MODEL, "raw": text,
           "predictions": extract_predictions(text), "usage": usage,
           "num_train": len(p.get("training", [])), "num_test": len(queries),
           "ground_truth": p.get("testing", []), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(rec, open(cache, "w"))
    return rec

def main():
    corpus = sys.argv[1]
    limit = 0; workers = 8
    if "--limit" in sys.argv: limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--workers" in sys.argv: workers = int(sys.argv[sys.argv.index("--workers") + 1])
    ddir = "data" if corpus == "wu" else "data_spa"
    rdir = "results" if corpus == "wu" else "results_spa"
    data = json.load(open(os.path.join(ROOT, ddir, "processed_dataset.json")))
    ids = sorted(data)[:limit] if limit else sorted(data)
    rawdir = os.path.join(ROOT, rdir, f"qwen_raw_{corpus}")
    os.makedirs(rawdir, exist_ok=True)
    done, lock, t0 = [], threading.Lock(), time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, pid, data[pid], corpus, rawdir): pid for pid in ids}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
                with lock: done.append(rec)
            except Exception as e:
                sys.stderr.write(f"FAIL {futs[f]}: {e}\n")
            if i % 20 == 0 or i == len(ids):
                el = time.time() - t0
                print(f"[{i}/{len(ids)}] {el/60:.1f} min elapsed, "
                      f"eta {(el/i*(len(ids)-i))/60:.1f} min", flush=True)
    out = os.path.join(ROOT, rdir, f"qwen_ic_predictions_{corpus}.json")
    json.dump({"model": MODEL, "base_url": BASE_URL, "n": len(done),
               "results": done}, open(out, "w"))
    print(f"saved {out} ({len(done)} participants)")

if __name__ == "__main__":
    main()
