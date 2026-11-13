# transformation_matrixs_fused = compute_language_agnostic_via_concat_svd(transformation_matrixs_fused_list, k)
    # # =============================== 低秩近似 ===============================
    # T = torch.tensor(np.array(transformation_matrixs_fused_list), dtype=torch.float32).to("cuda")  # (N, L, d, d)
    # N, L, d, _ = T.shape

    # # Store low-rank approximations per language per layer
    # T_lowrank = torch.zeros_like(T)  # (N, L, d, d)

    # for lang in range(N):
    #     for layer in range(LAYERS):
    #         M = T[lang, layer, :, :]  # (d, d)
    #         try:
    #             U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    #         except RuntimeError:
    #             # Fallback to CPU if needed (unlikely)
    #             M_cpu = M.cpu()
    #             U, S, Vt = torch.linalg.svd(M_cpu, full_matrices=False)
    #             U, S, Vt = U.cuda(), S.cuda(), Vt.cuda()

    #         # # Keep top-k
    #         k_use = min(k, S.size(0))

    #         # >>> 动态计算 k_use based on energy <<<
    #         # energy_threshold = k
    #         # total_energy = torch.sum(S)
    #         # if total_energy > 0:
    #         #     cumsum_energy = torch.cumsum(S, dim=0)
    #         #     k_use = int(torch.searchsorted(cumsum_energy, energy_threshold * total_energy).item()) + 1
    #         #     k_use = min(k_use, S.size(0))
    #         # else:
    #         #     k_use = 1  # fallback
    #         # >>> NOT ADD SIGN ALIGNMENT HERE <<<
    #         # Reconstruct: U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    #         # logger.info(f"adding sign alignment")
    #         # S_k = S[:k_use]
    #         # U_k = U[:, :k_use]
    #         # Vt_k = Vt[:k_use, :]
    #         # M_lowrank = U_k @ torch.diag(S_k) @ Vt_k
    #         # T_lowrank[lang, layer, :, :] = M_lowrank

    #         # # >>> ADD SIGN ALIGNMENT HERE <<<
    #         signs = torch.sign(U[0, :k_use])      # Use first row as reference
    #         signs[signs == 0] = 1                 # Avoid zero
    #         U = U[:, :k_use] * signs              # (d, k)
    #         Vt = Vt[:k_use, :] * signs.unsqueeze(1)  # (k, d)
    #         S = S[:k_use]                         # (k,)

    #         M_lowrank = U @ torch.diag(S) @ Vt
    #         T_lowrank[lang, layer, :, :] = M_lowrank


    # # Now average over languages (after low-rank approximation)
    # transformation_matrixs_fused = T_lowrank.mean(dim=0).cpu().numpy()  # (L, d, d)

    # # =============================== 低秩近似 ===============================
    # # =============================== PCA ===============================

    # # Convert to GPU tensor: shape (num_langs, LAYERS, d, d)
    # T = torch.tensor(np.array(transformation_matrixs_fused_list), dtype=torch.float32, device="cuda")
    # N_langs, _, d, _ = T.shape

    # # Will hold the final shared projection matrix per layer: (LAYERS, d, d)
    # transformation_matrixs_fused = []

    # # Process each layer independently
    # for layer in range(LAYERS):
    #     # Step 1: Gather all language matrices for this layer → (N_langs, d*d)
    #     X = T[:, layer, :, :].view(N_langs, -1)  # (N, D), D = d*d

    #     # Step 2: Center the data (PCA requires this!)
    #     mean_X = X.mean(dim=0, keepdim=True)      # (1, D)
    #     X_centered = X - mean_X                   # (N, D)

    #     # Step 3: Perform SVD on centered data → this is PCA
    #     U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    #     # Step 4: Reconstruct using top-k principal components
    #     k_use = min(k, S.size(0))
    #     # Project and reconstruct: X_denoised = mean_X + U_k @ diag(S_k) @ Vt_k
    #     X_recon = mean_X + torch.mm(U[:, :k_use] * S[:k_use], Vt[:k_use, :])  # (N, D)

    #     # Step 5: Average over languages in the denoised space
    #     shared_vec = X_recon.mean(dim=0)          # (D,)
    #     shared_matrix = shared_vec.view(d, d).cpu().numpy()  # back to CPU for compatibility
    #     transformation_matrixs_fused.append(shared_matrix)

    # transformation_matrixs_fused = np.stack(transformation_matrixs_fused, axis=0)  # (LAYERS, d, d)


    # # =============================== PCA ===============================

    # transformation_matrixs_fused = np.array(transformation_matrixs_fused)  # (L, d, d)


    # ipdb.set_trace()