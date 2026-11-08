"""Fixed-seed selection of 128 train and 32 development HQ rows."""
import json, os, hashlib
from pathlib import Path
import pyarrow.parquet as pq
from ricepo.artifacts import atomic_json, file_sha256
from ricepo.data import canonical_query, join_reasonir_hq
from ricepo.manifest import build_manifests

root = Path(os.environ["RICE_PROJECT"])
target = root/"assets/exports/biology-batch-v1"
target.mkdir()
old = json.loads((root/"assets/exports/biology-smoke-v2/preparation-plan.json").read_text())
spec = old["domains"]["biology"]
corpus = {json.loads(line)["id"]:"" for line in Path(spec["bright_documents_jsonl"]["path"]).open()}
test_texts = {canonical_query(json.loads(line)["query"])
              for line in Path(spec["bright_examples_jsonl"]["path"]).open()}
hq = root/"assets/frozen/reasonir_hq/0275f825929b206d4ead23d34b4f8a50d4eddbc8/hq/train-00000-of-00001.parquet"
rows, seen, index, rejected = [], set(test_texts), 0, {}
for batch in pq.ParquetFile(hq).iter_batches(batch_size=8192):
    for row in batch.to_pylist():
        source = index
        index += 1
        try:
            query, positive = join_reasonir_hq(row,corpus)
        except (ValueError,KeyError,TypeError) as exc:
            key = type(exc).__name__
            rejected[key] = rejected.get(key,0)+1
            continue
        normal = canonical_query(query)
        if normal in seen:
            rejected["duplicate_or_test"] = rejected.get("duplicate_or_test",0)+1
            continue
        seen.add(normal)
        rank = hashlib.sha256(json.dumps([20261105,source],separators=(",",":")).encode()).hexdigest()
        rows.append((rank,source,row))
rows.sort()
selected = rows[:160]
if len(selected) < 2:
    raise ValueError("insufficient disjoint eligible Biology rows")
dev_count = min(32,max(1,len(selected)//5))
train_count = len(selected)-dev_count
export = target/"reasonir-selected.jsonl"
selection = target/"selection.jsonl"
with export.open("x",encoding="utf8") as f, selection.open("x",encoding="utf8") as s:
    for i,(_,source,row) in enumerate(selected):
        f.write(json.dumps(row,ensure_ascii=False)+"\n")
        s.write(json.dumps(dict(split="train" if i<train_count else "development", domain="biology",
                 query_id=f"reasonir-hq-row-{source}",source_row_index=i))+"\n")
atomic_json(target/"source-rows.json",dict(seed=20261105,eligible=len(rows),examined=index,rejected=rejected,
               original_rows=[source for _,source,_ in selected],upstream_parquet_sha256=file_sha256(hq)))
spec = {**spec, "reasonir_hq_jsonl":{"path":str(export),"sha256":file_sha256(export)}}
plan = {**old, "domains":{"biology":spec},"selection_jsonl":{"path":str(selection),"sha256":file_sha256(selection)},
 "selection_provenance":{split:dict(rule="ascending SHA256([20261105,original HQ row]); train then development",
          filters=["all positives map to full Biology corpus","canonical text distinct within and across splits",
                   "exclude official Biology test query text"],seed=20261105) for split in ("train","development")}}
atomic_json(target/"preparation-plan.json",plan)
bundle = root/"datasets/biology-batch-bundle-v1"
receipt = build_manifests(target/"preparation-plan.json",bundle)
print(json.dumps(dict(counts=receipt["counts"],receipt_sha256=file_sha256(bundle/"receipt.json"))))
