"""Offline native-output classification and bounded local asset inventory."""
from collections import Counter, defaultdict
import os, re, json
from pathlib import Path
from transformers import AutoTokenizer
from ricepo.artifacts import atomic_json, file_sha256, require_slurm
from ricepo.state import read_json
from ricepo.formats import action_closed

def main():
    require_slurm()
    r=Path(os.environ["RICE_PROJECT"])
    out=r/"reports"/("protocol-audit-"+os.environ["SLURM_JOB_ID"])
    out.mkdir()
    base=Path(os.environ["RICE_FIXED_DEV_RUN"])
    prior=Path(os.environ["RICE_PRIOR_PROMPT_RUN"])
    dep=read_json(Path(os.environ["RICE_SOURCE_CONFIG"]))["deployment"]
    tok=AutoTokenizer.from_pretrained(dep["actor_model_path"],local_files_only=True)
    rows=[]
    for row in read_json(base/"summary.json")["records"]:
        if row["status"]!="failed":
            continue
        run=prior if row.get("archived") else base
        hits=[]
        for path in run.glob("attempts/*/events/*.json"):
            value=read_json(path)
            if (value.get("kind")=="raw_completion" and value.get("query_id")==row["query"]
                    and value.get("depth")==row["depth"]):
                hits.append((path,value))
        path, value=hits[0] if len(hits)==1 else (_ for _ in ()).throw(ValueError("ambiguous raw failure"))
        ids=value["token_ids"]
        end=len(ids)
        while end and ids[end-1] in (tok.eos_token_id,tok.pad_token_id):
            end-=1
        text=tok.decode(ids[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        tags=Counter(re.findall(r"</?(?:think|summary|tool_response)>",text))
        labels=[]
        if tags["<think>"]:
            labels.append("repeated_prefilled_think")
        for tag,label in (("</think>","missing_think_close"),("<summary>","missing_summary_open"),
                          ("</summary>","missing_summary_close")):
            if not tags[tag]:
                labels.append(label)
        stack=["think"]
        nested=False
        mismatched=False
        for match in re.finditer(r"<(/?)(think|summary)>",text):
            close,name=match.groups()
            if not close:
                nested |= bool(stack)
                stack.append(name)
            elif stack and stack[-1]==name:
                stack.pop()
            else:
                mismatched=True
        if nested:
            labels.append("nested_tags")
        if any(tags[tag]>1 for tag in ("</think>","<summary>","</summary>")):
            labels.append("duplicate_tags")
        if mismatched:
            labels.append("mismatched_tag_order")
        if value["finish_reason"]=="eos" and not action_closed(text):
            labels.append("premature_eos")
        if row["category"]=="generation_length":
            labels.append("generation_length")
        if not labels:
            labels.append("other_structure")
        evidence=dict(source_sha256=file_sha256(path),source_relative_path=str(path.relative_to(r)),
                      request_id=value.get("request_id"),policy_version=value.get("policy_version"),
                      prefix_sha256=value.get("prefix_sha256"),seed=value.get("seed"),
                      native_ids=ids,decoded_text=text,
                      full_decoded_text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
                      original_category=row["category"],depth=row["depth"],labels=labels,
                      tag_counts=dict(tags),finish_reason=value["finish_reason"])
        atomic_json(out/"evidence"/(str(len(rows))+".json"),evidence)
        rows.append({k:v for k,v in evidence.items() if k not in (
            "native_ids","decoded_text","full_decoded_text","source_relative_path","request_id","policy_version","seed")})
    counts=Counter()
    by_depth=defaultdict(Counter)
    for row in rows:
        counts.update(row["labels"])
        by_depth[str(row["depth"])].update(row["labels"])
    assets=[]
    for root in (r.parents[1]/"models",r/"assets",r.parents[2]/".cache/huggingface/hub"):
        if not root.exists():
            continue
        for current,dirs,files in os.walk(root,followlinks=False):
            relative=Path(current).relative_to(root)
            if len(relative.parts)>5:
                dirs[:]=[]
                continue
            if "qwen3-4b-thinking-2507" in str(current).lower():
                assets.append(dict(directory_name=Path(current).name,config="config.json" in files,
                                   weight_files=sum(f.endswith((".safetensors",".bin")) for f in files)))
    atomic_json(out/"summary.json",dict(failed_queries=len(rows),multi_label_counts=dict(counts),
        by_depth={k:dict(v) for k,v in by_depth.items()},cases=rows,qwen3_4b_assets=assets,
        asset_scan_scope="model directory, project assets, default HF cache; max depth six; no symlinks"))
    print(json.dumps(dict(failed_queries=len(rows),counts=dict(counts),
                         by_depth={k:dict(v) for k,v in by_depth.items()},assets=assets)))


if __name__ == "__main__":
    main()
