import torch
import numpy as np
from typing import List
from collections import defaultdict
import re
from util import nethook
from util.rome.util import Trace, TraceDict
from transformers import GenerationConfig
import ipdb

punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']


def get_cache_path(model_name, lang):
    model_name = model_name.replace("/", "_")
    return f"cache/{model_name}/transformation_matrixs_{lang}.pkl"

def rect_mask_update(dW, W_ref, topk=0.4, eps=1e-8):
    """
    dW: (n, m) update matrix for one language (ΔW)
    W_ref: (n, m) original/base weight matrix (W)
    topk: keep top-k% by relative change |ΔW/W|
    """
    # ipdb.set_trace()
    rel = np.abs(dW) / (np.abs(W_ref) + eps)          # δ = |ΔW/W|
    flat = rel.reshape(-1)
    k = int(np.ceil(topk * flat.numel()))

    # threshold for top-k%
    thresh = np.partition(flat, -k)[-k]
    mask = rel <= thresh
    return dW * mask

# sigmas = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


# def compute_language_agnostic_projections(
# def compute_language_agnostic_via_svd_lowrank(
#     transformation_matrixs_fused_list,  # list of list: [lang][layer] -> (d, d) np.ndarray
#     k=64,
#     device="cuda"
# ):
#     """
#     Compute language-agnostic projection matrices via SVD low-rank approximation
#     with sign alignment and factor-wise averaging.

#     Args:
#         transformation_matrixs_fused_list: List[List[np.ndarray]] 
#             shape: [num_langs][num_layers], each is (d, d)
#         k: int, rank for low-rank approximation
#         device: str

#     Returns:
#         transformation_matrixs_fused: np.ndarray of shape (num_layers, d, d)
#     """
#     num_langs = len(transformation_matrixs_fused_list)
#     num_layers = len(transformation_matrixs_fused_list[0])
#     d = transformation_matrixs_fused_list[0][0].shape[0]

#     # Prepare lists to collect U, S, Vt per (lang, layer)
#     U_all = []   # will be list of list: [lang][layer] -> (d, k_use)
#     S_all = []   # [lang][layer] -> (k_use,)
#     Vt_all = []  # [lang][layer] -> (k_use, d)

#     for lang in range(num_langs):
#         U_lang = []
#         S_lang = []
#         Vt_lang = []
#         for layer in range(num_layers):
#             M_np = transformation_matrixs_fused_list[lang][layer]
#             M = torch.tensor(M_np, dtype=torch.float32, device=device)

#             try:
#                 U, S, Vt = torch.linalg.svd(M, full_matrices=False)
#             except RuntimeError:
#                 # Fallback to CPU
#                 M_cpu = M.cpu()
#                 U, S, Vt = torch.linalg.svd(M_cpu, full_matrices=False)
#                 U, S, Vt = U.to(device), S.to(device), Vt.to(device)

#             k_use = min(k, S.size(0))

#             # === SIGN ALIGNMENT (critical!) ===
#             # Use first row of U as reference to fix sign ambiguity
#             # >>> NOT ADD SIGN ALIGNMENT HERE <<<
#             # ipdb.set_trace()
#             U = U[:, :k_use]         # (d, k)
#             Vt = Vt[:k_use, :]        # (k, d)
#             S = S[:k_use]                                  # (k,)

#             U_lang.append(U)
#             S_lang.append(S)
#             Vt_lang.append(Vt)

#         U_all.append(U_lang)
#         S_all.append(S_lang)
#         Vt_all.append(Vt_lang)

#     # Now average over languages for each layer
#     fused_matrices = []

#     for layer in range(num_layers):
#         # Stack across languages: (num_langs, d, k) etc.
#         U_stack = torch.stack([U_all[lang][layer] for lang in range(num_langs)], dim=0)  # (N, d, k)
#         S_stack = torch.stack([S_all[lang][layer] for lang in range(num_langs)], dim=0)  # (N, k)
#         Vt_stack = torch.stack([Vt_all[lang][layer] for lang in range(num_langs)], dim=0)  # (N, k, d)

#         # Average
#         U_avg = U_stack.mean(dim=0)      # (d, k)
#         S_avg = S_stack.mean(dim=0)      # (k,)
#         Vt_avg = Vt_stack.mean(dim=0)    # (k, d)

#         # Reconstruct full matrix
#         P_avg = U_avg @ torch.diag(S_avg) @ Vt_avg  # (d, d)
#         fused_matrices.append(P_avg.cpu().numpy())

#     # Stack into (num_layers, d, d)
#     transformation_matrixs_fused = np.stack(fused_matrices, axis=0)
#     return transformation_matrixs_fused
# compute_language_agnostic_via_svd_2
def compute_language_agnostic_via_svd_2(
    transformation_matrixs_fused_list,  # list of list: [lang][layer] -> (d, d) np.ndarray
    k=64,
    device="cuda"
):
    """
    Compute language-agnostic projection matrices by:
    1) averaging matrices across languages (per layer),
    2) applying SVD on the mean matrix,
    3) reconstructing with top-k singular components.

    Args:
        transformation_matrixs_fused_list: List[List[np.ndarray]]
            shape: [num_langs][num_layers], each is (d, d)
        k: int, rank for low-rank approximation
        device: str

    Returns:
        transformation_matrixs_fused: np.ndarray of shape (num_layers, d, d)
    """
    num_langs = len(transformation_matrixs_fused_list)
    num_layers = len(transformation_matrixs_fused_list[0])
    d = transformation_matrixs_fused_list[0][0].shape[0]

    fused_matrices = []

    for layer in range(num_layers):
        # 1) mean matrix across languages for this layer: (d, d)
        mean_np = np.mean(
            [transformation_matrixs_fused_list[lang][layer] for lang in range(num_langs)],
            axis=0
        )

        M = torch.tensor(mean_np, dtype=torch.float32, device=device)

        # 2) SVD on mean matrix
        try:
            U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU
            U, S, Vt = torch.linalg.svd(M.cpu(), full_matrices=False)
            U, S, Vt = U.to(device), S.to(device), Vt.to(device)

        k_use = min(k, S.size(0))

        # 3) take top-k and reconstruct
        U_k = U[:, :k_use]          # (d, k)
        S_k = S[:k_use]             # (k,)
        Vt_k = Vt[:k_use, :]        # (k, d)

        P_k = U_k @ torch.diag(S_k) @ Vt_k  # (d, d)
        fused_matrices.append(P_k.cpu().numpy())

    transformation_matrixs_fused = np.stack(fused_matrices, axis=0)  # (num_layers, d, d)
    return transformation_matrixs_fused



def compute_language_agnostic_via_svd_lowrank(
    transformation_matrixs_fused_list,  # list of list: [lang][layer] -> (d, d) np.ndarray
    k=64,
    device="cuda"
):
    """
    Compute language-agnostic projection matrices by:
    1) averaging matrices across languages (per layer),
    2) applying SVD on the mean matrix,
    3) reconstructing with top-k singular components.

    Args:
        transformation_matrixs_fused_list: List[List[np.ndarray]]
            shape: [num_langs][num_layers], each is (d, d)
        k: int, rank for low-rank approximation
        device: str

    Returns:
        transformation_matrixs_fused: np.ndarray of shape (num_layers, d, d)
    """
    num_langs = len(transformation_matrixs_fused_list)
    num_layers = len(transformation_matrixs_fused_list[0])
    d = transformation_matrixs_fused_list[0][0].shape[0]

    fused_matrices = []
    all_fused_matrices = []
    # null_space_matrices = []
    # threshold = 1e-1
    for layer in range(num_layers):
        # 1) mean matrix across languages for this layer: (d, d)
        mean_np = np.mean(
            [transformation_matrixs_fused_list[lang][layer] for lang in range(num_langs)],
            axis=0
        )

        M = torch.tensor(mean_np, dtype=torch.float32, device=device)

        # 2) SVD on mean matrix
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)

        k_use = min(k, S.size(0))

        # 3) take top-k and reconstruct
        U_k = U[:, :k_use]          # (d, k)
        S_k = S[:k_use]             # (k,)
        Vt_k = Vt[:k_use, :]        # (k, d)

        # small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
        P_k = U_k @ torch.diag(S_k) @ Vt_k  # (d, d)
        # ipdb.set_trace()
        # null_space_projection = U[:, small_singular_indices] @ U[:, small_singular_indices].T
        # P_specific = null_space_projection@(M - P_k)
        P_specific = M - P_k

        fused_matrices.append(P_k.cpu().numpy())
        all_fused_matrices.append(P_specific.cpu().numpy())
        # null_space_matrices.append(P_specific)

    transformation_matrixs_fused = np.stack(fused_matrices, axis=0)  # (num_layers, d, d)
    all_fused_matrixs = np.stack(all_fused_matrices, axis=0)  # (num_langs, num_layers, d, d)
    # null_space_projection = np.stack(null_space_matrices, axis=0)  # (num_layers, d, d)
    # delta_fused_matrixs = rect_mask_update(all_fused_matrixs, transformation_matrixs_fused, topk=0.1)
    # delta_fused_matrixs = all_fused_matrixs - transformation_matrixs_fused
    # transformation_matrixs_fused = transformation_matrixs_fused + all_fused_matrixs
    
    return (transformation_matrixs_fused,all_fused_matrixs)

# def compute_language_agnostic_via_svd_2(
#     transformation_matrixs_fused_list,  # list of list: [lang][layer] -> (d, d) np.ndarray
#     k=64,
#     device="cuda"
# ):
#     transformation_matrixs_fused = transformation_matrixs_fused_list[0]
#     for lang_id in range(1,len(transformation_matrixs_fused_list)):
#         transformation_matrixs_fused += rect_mask_update(transformation_matrixs_fused_list[lang_id], transformation_matrixs_fused, topk=0.3)
#     transformation_matrixs_fused /= len(transformation_matrixs_fused_list)
#     return transformation_matrixs_fused

# def compute_language_agnostic_via_svd_2(
#     transformation_matrixs_fused_list,  # list of list: [lang][layer] -> (d, d) np.ndarray
#     k=64,
#     device="cuda"
# ):
#     num_langs = len(transformation_matrixs_fused_list)
#     num_layers = len(transformation_matrixs_fused_list[0])
#     d = transformation_matrixs_fused_list[0][0].shape[0]

#     fused_matrices = []

#     for layer in range(num_layers):
#         # 1) mean matrix across languages for this layer: (d, d)
#         mean_np = np.mean(
#             [transformation_matrixs_fused_list[lang][layer] for lang in range(num_langs)],
#             axis=0
#         )

#         M = torch.tensor(mean_np, dtype=torch.float32, device=device)

#         # 2) SVD on mean matrix
#         U, S, Vt = torch.linalg.svd(M, full_matrices=False)

#         k_use = min(k, S.size(0))

#         # 3) take top-k and reconstruct
#         U_k = U[:, :k_use]          # (d, k)
#         S_k = S[:k_use]             # (k,)
#         Vt_k = Vt[:k_use, :]        # (k, d)

#         P_k = U_k @ torch.diag(S_k) @ Vt_k  # (d, d)

#         fused_matrices.append(P_k.cpu().numpy())

#     transformation_matrixs_fused = np.stack(fused_matrices, axis=0)  # (num_layers, d, d)
#     transformation_matrixs_all = np.stack(transformation_matrixs_fused_list, axis=0)  # (num_langs, num_layers, d, d)

#     delta_matrixs = transformation_matrixs_all - transformation_matrixs_fused[None, :, :, :]

#     delta_fused_matrices = []

#     for layer in range(num_layers):
#         # 1) mean matrix across languages for this layer: (d, d)
#         mean_np = np.mean(
#             [delta_matrixs[lang][layer] for lang in range(num_langs)],
#             axis=0
#         )

#         M = torch.tensor(mean_np, dtype=torch.float32, device=device)

#         # 2) SVD on mean matrix
#         U, S, Vt = torch.linalg.svd(M, full_matrices=False)

#         k_use = min(k, S.size(0))

#         # 3) take top-k and reconstruct
#         U_k = U[:, :k_use]          # (d, k)
#         S_k = S[:k_use]             # (k,)
#         Vt_k = Vt[:k_use, :]        # (k, d)

#         P_k = U_k @ torch.diag(S_k) @ Vt_k  # (d, d)

#         delta_fused_matrices.append(P_k.cpu().numpy())

#     delta_fused_matrices = np.stack(delta_fused_matrices, axis=0)  # (num_layers, d, d)

#     transformation_matrixs_fused = transformation_matrixs_fused + delta_fused_matrices

#     return transformation_matrixs_fused




def compute_language_agnostic_via_concat_svd(
    transformation_matrixs_fused_list: List[List[np.ndarray]],
    k: int = 64,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute language-agnostic projection matrices by:
      1. Low-rank SVD per language per layer
      2. Sign alignment of singular vectors
      3. Concatenate U and V across languages
      4. Extract shared subspace via second SVD
      5. Reconstruct fused projection matrix

    Args:
        transformation_matrixs_fused_list: List of list of (d, d) np.ndarray
            [lang_idx][layer_idx] -> (d, d)
        k: rank for low-rank approximation (per language)
        device: 'cuda' or 'cpu'

    Returns:
        transformation_matrixs_fused: np.ndarray of shape (num_layers, d, d)
    """
    num_langs = len(transformation_matrixs_fused_list)
    num_layers = len(transformation_matrixs_fused_list[0])
    d = transformation_matrixs_fused_list[0][0].shape[0]

    fused_matrices = []

    for layer in range(num_layers):
        U_langs = []   # list of (d, k_use)
        V_langs = []   # list of (d, k_use)
        S_langs = []   # list of (k_use,)

        # Step 1: SVD per language for this layer
        for lang in range(num_langs):
            M_np = transformation_matrixs_fused_list[lang][layer]
            M = torch.tensor(M_np, dtype=torch.float32, device=device)

            try:
                U, S, Vt = torch.linalg.svd(M, full_matrices=False)
            except RuntimeError:
                M_cpu = M.cpu()
                U, S, Vt = torch.linalg.svd(M_cpu, full_matrices=False)
                U, S, Vt = U.to(device), S.to(device), Vt.to(device)

            k_use = min(k, S.size(0))

            # === Sign alignment using first row of U ===
            signs = torch.sign(U[0, :k_use])
            signs[signs == 0] = 1
            U = U[:, :k_use] * signs.unsqueeze(0)          # (d, k)
            V = Vt[:k_use, :].t() * signs.unsqueeze(0)     # (d, k), note: V = Vt.T
            S = S[:k_use]                                  # (k,)

            U_langs.append(U)
            V_langs.append(V)
            S_langs.append(S)

        # Step 2: Concatenate across languages
        U_cat = torch.cat(U_langs, dim=1)  # (d, num_langs * k_use)
        V_cat = torch.cat(V_langs, dim=1)  # (d, num_langs * k_use)

        # Step 3: Extract shared subspace via SVD
        # We only need top-k directions that capture cross-lingual consensus
        k_final = min(k, U_cat.size(1))
        U_shared, _, _ = torch.linalg.svd(U_cat, full_matrices=False)
        V_shared, _, _ = torch.linalg.svd(V_cat, full_matrices=False)

        U_shared = U_shared[:, :k_final]  # (d, k_final)
        V_shared = V_shared[:, :k_final]  # (d, k_final)

        # Step 4: Average singular values
        S_avg = torch.stack(S_langs).mean(dim=0)  # (k_use,)
        if S_avg.size(0) > k_final:
            S_avg = S_avg[:k_final]
        elif S_avg.size(0) < k_final:
            # Pad with zeros if needed (rare)
            S_avg = torch.cat([S_avg, torch.zeros(k_final - S_avg.size(0), device=device)])

        # Step 5: Reconstruct language-agnostic projection
        P_agnostic = U_shared @ torch.diag(S_avg) @ V_shared.t()  # (d, d)
        fused_matrices.append(P_agnostic.cpu().numpy())

    # Stack layers into (num_layers, d, d)
    return np.stack(fused_matrices, axis=0)

def compute_mean_transformation_matrix(
    transformation_matrixs_fused_list,
):
    """
    Compute mean transformation matrix from a list of transformation matrices.
    transformation_matrixs_fused_list: List[List[np.ndarray]]
            shape: [num_langs][num_layers], each is (d, d) np.ndarray
    """
    return np.mean(transformation_matrixs_fused_list, axis=0)


def trace_with_patch_generative(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    new_mlp_up,
    question,
    answer,
    mt,  # ModelAndTokenizer instance
):
    # ipdb.set_trace()
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = untuple(x)
        idxxx = h.shape[1] - 1
        if idxxx == patch_spec[layer][0]:
            for t in patch_spec[layer]:
                layer_match = re.search(r"(?:transformer\.h|model\.layers|gpt_neox\.layers)\.(\d+)", layer)
                if layer_match is None:
                    raise ValueError(f"Cannot parse layer index from {layer}")
                ly = int(layer_match.group(1))
                h[0][t] = torch.tensor(new_mlp_up)[ly]
        return h
    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        
        input_ids_w_attnmask = mt.tokenizer(question,padding=True,return_tensors="pt",).to('cuda')
        output_ids = model.generate(input_ids=input_ids_w_attnmask.input_ids,attention_mask=input_ids_w_attnmask.attention_mask,
                    generation_config=GenerationConfig(
                        max_new_tokens=600,
                        do_sample=False,
                        temperature=0.0,  
                    ),
                ).tolist()
        real_output_ids = [output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)]
        r = mt.tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]
   
        return r.strip()


def trace_with_patch_discrimination(
    model,  
    inp,  #
    states_to_patch,  
    new_mlp_up,
    question,
    answer,
    mt,  # ModelAndTokenizer instance
):
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = untuple(x)
        for t in patch_spec[layer]:
            layer_match = re.search(r"(?:transformer\.h|model\.layers|gpt_neox\.layers)\.(\d+)", layer)
            if layer_match is None:
                raise ValueError(f"Cannot parse layer index from {layer}")
            ly = int(layer_match.group(1))
            h[0][t] = torch.tensor(new_mlp_up)[ly]
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:
        # ipdb.set_trace()
        target_ids = mt.tokenizer(' '+answer + '</s>', return_tensors='pt')['input_ids'].to('cuda')

        outputs = model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()
        # ipdb.set_trace()
        textual_ans = mt.tokenizer.decode(ans_idss, skip_special_tokens=True)
        for p in punc:
            textual_ans = textual_ans.replace(p, '')
        
        return textual_ans.strip()
