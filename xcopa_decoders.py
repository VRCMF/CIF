import re
from collections import defaultdict

import torch

from util import nethook


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _layer_index(layer):
    match = re.search(r"(?:transformer\.h|model\.layers|gpt_neox\.layers)\.(\d+)", layer)
    if match is None:
        raise ValueError(f"Cannot parse layer index from {layer}")
    return int(match.group(1))


def _candidate_token_ids(tokenizer, label):
    candidates = []
    for text in (f" {label}", label, f'"{label}"', f' "{label}"'):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            candidates.append(ids[0])
    return sorted(set(candidates))


def trace_with_patch_choice_logits(model, inp, states_to_patch, new_mlp_up, mt):
    patch_spec = defaultdict(list)
    for token_index, layer in states_to_patch:
        patch_spec[layer].append(token_index)

    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = _untuple(x)
        update = torch.as_tensor(new_mlp_up, dtype=h.dtype, device=h.device)
        for token_index in patch_spec[layer]:
            h[0][token_index] = update[_layer_index(layer)]
        return h

    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ):
        outputs = model(**inp)

    logits = outputs.logits[0, -1]
    a_ids = _candidate_token_ids(mt.tokenizer, "A")
    b_ids = _candidate_token_ids(mt.tokenizer, "B")
    a_score = torch.logsumexp(logits[a_ids], dim=0)
    b_score = torch.logsumexp(logits[b_ids], dim=0)
    return "A" if a_score >= b_score else "B"
