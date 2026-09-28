"""Build chunk-level dataset from CUADv1.json.

Every contract is cut with the SAME chunker used at inference time, and each chunk
gets a label + a negative bucket:
  pos          overlaps a Liquidated Damages span
  hard_neg     overlaps a span of another CUAD category
  boiler_neg   overlaps no annotation at all (the "real contract is mostly boilerplate" case)

Usage: python src/prepare_data.py --cuad data/raw/CUADv1.json [--test_json data/raw/test.json]
"""
import argparse, json, os
import pandas as pd
from chunker import chunk_text

LD = "Liquidated Damages"

def load_cuad(path):
    d = json.load(open(path))
    docs = []
    for doc in d["data"]:
        p = doc["paragraphs"][0]
        spans = {}
        for qa in p["qas"]:
            cat = qa["id"].split("__")[-1]
            spans[cat] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
        docs.append({"contract_id": doc["title"], "text": p["context"], "spans": spans})
    return docs

def overlap(a, b, x, y):
    return max(0, min(b, y) - max(a, x))

def label_chunks(doc):
    ld = doc["spans"].get(LD, [])
    other = [s for c, ss in doc["spans"].items() if c != LD for s in ss]
    rows = []
    for i, (a, b, txt) in enumerate(chunk_text(doc["text"])):
        # positive if the chunk holds >=50% of an LD span, or LD text is >=50% of the chunk
        pos = any(overlap(a, b, x, y) >= 0.5 * min(y - x, b - a) for x, y in ld)
        hard = (not pos) and any(overlap(a, b, x, y) > 0 for x, y in other)
        touches_ld = [j for j, (x, y) in enumerate(ld) if overlap(a, b, x, y) > 0]
        rows.append({"contract_id": doc["contract_id"], "chunk_id": i, "start": a, "end": b,
                     "text": txt, "label": int(pos),
                     "bucket": "pos" if pos else ("hard_neg" if hard else "boiler_neg"),
                     "ld_span_ids": ",".join(map(str, touches_ld))})
    return rows

def span_table(docs):
    return pd.DataFrame([{"contract_id": d["contract_id"], "span_id": j, "start": x, "end": y,
                          "text": d["text"][x:y]} for d in docs for j, (x, y) in enumerate(d["spans"].get(LD, []))])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad", default="data/raw/CUADv1.json")
    ap.add_argument("--test_json", default=None, help="official CUAD test.json -> fixes the test contracts")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    docs = load_cuad(args.cuad)
    chunks = pd.DataFrame([r for d in docs for r in label_chunks(d)])
    spans = span_table(docs)
    contracts = pd.DataFrame([{"contract_id": d["contract_id"], "n_chars": len(d["text"]),
                               "has_ld": int(bool(d["spans"].get(LD)))} for d in docs])
    if args.test_json:
        test_ids = {x["title"] for x in json.load(open(args.test_json))["data"]}
        contracts["split"] = contracts.contract_id.map(lambda c: "test" if c in test_ids else "train")
    import os; os.makedirs(args.out, exist_ok=True)
    chunks.to_csv(f"{args.out}/chunks.csv.gz", index=False); spans.to_csv(f"{args.out}/ld_spans.csv", index=False)
    contracts.to_csv(f"{args.out}/contracts.csv", index=False)
    print(chunks.bucket.value_counts()); print("contracts", len(contracts), "with LD", contracts.has_ld.sum(), "spans", len(spans))
