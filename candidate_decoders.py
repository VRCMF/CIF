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


def trace_with_patch_candidate_logprobs(
    model,
    prompt,
    candidates,
    states_to_patch,
    new_mlp_up,
    mt,
    normalize=True,
):
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

    prompt_ids = mt.tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("candidate decoder received an empty prompt")

    scores = []
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ):
        for candidate in candidates:
            text = prompt + " " + candidate
            input_ids = mt.tokenizer.encode(text, return_tensors="pt").to(model.device)
            candidate_ids = input_ids[0, len(prompt_ids):]
            if candidate_ids.numel() == 0:
                scores.append(float("-inf"))
                continue
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
            start = len(prompt_ids) - 1
            end = start + candidate_ids.numel()
            log_probs = torch.log_softmax(logits[start:end], dim=-1)
            token_scores = log_probs.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)
            score = token_scores.mean() if normalize else token_scores.sum()
            scores.append(float(score.detach().cpu()))

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx]
