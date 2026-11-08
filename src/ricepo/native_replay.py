"""Explicit recovery reuse of paid native completions from identical behavior.

Source/data/weights binding is checked before indexing. No decoding, resampling,
or merging mismatched policies. A replay is not a new independent sample.
Only complete saved raw responses are reusable; tool calls are rerun normally.
"""
import json
from pathlib import Path
from .artifacts import file_sha256, require_cluster_storage
from .audit import fingerprint


class SavedNativeCompletions:
    def __init__(self,sources,binding):
        if not isinstance(sources,list) or not sources:
            raise ValueError("explicit nonempty replay source list required")
        self.records={}
        # Source code and run directories may differ for a documented parser
        # repair. All scientific/behavior/asset identities must remain equal.
        keys=("config_sha256","recipe_fingerprint","dataset_receipt_sha256","manifest_sha256",
              "model_sha256","retriever_sha256","prompt_sha256","behavior")
        for value in sources:
            root=require_cluster_storage(value)
            old=json.loads((root/"binding.json").read_text())
            if any(k not in old or k not in binding or old[k]!=binding[k] for k in keys):
                raise ValueError("saved native behavior, recipe or assets differ from current run")
            for path in sorted(root.glob("attempts/*/events/*.json")):
                item=json.loads(path.read_text())
                if item.get("kind")!="raw_completion" or item.get("recovery_replay"):
                    continue
                prefix=tuple(item["prefix_token_ids"])
                if fingerprint(prefix)!=item["prefix_sha256"]:
                    raise ValueError("saved native prefix fingerprint changed")
                key=(item["policy_version"],item["seed"],item["prefix_sha256"])
                record=dict(token_ids=tuple(item["token_ids"]),finish_reason=item["finish_reason"],
                    source=dict(event_sha256=file_sha256(path),source_binding_sha256=file_sha256(root/"binding.json"),
                                source_code_sha256=old["source_sha256"]))
                if key in self.records and any(self.records[key][k]!=record[k] for k in ("token_ids","finish_reason")):
                    raise ValueError("contradictory saved completions for identical request")
                self.records[key]=record

    def __call__(self,prefix,seed,version):
        return self.records.get((version,seed,fingerprint(tuple(prefix))))
