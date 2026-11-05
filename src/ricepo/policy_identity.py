"""Identify the live, single-process sampling policy at each weight transition."""
from hashlib import sha256
import json
from .audit import fingerprint, operation


def identify(model, behavior):
    import torch
    if model.training:
        raise ValueError("behavior identity requires eval mode")
    digest = sha256()
    with operation("behavior_weight_identity"), torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            header = json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(",",":")).encode()
            digest.update(len(header).to_bytes(8,"big"))
            digest.update(header)
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            for block in raw.split(4*1024*1024):
                digest.update(block.cpu().numpy().tobytes())
    identity = {"state_dict_sha256":digest.hexdigest(),"behavior":behavior}
    return fingerprint(identity),identity
