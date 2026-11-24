// Body of the KV-block loop in flash_attn.comp, included once per value of FA_KV_CHECK.
        if (MASK_ENABLE) {
            if (USE_MASK_OPT && mask_opt_idx != j / 16) {
                mask_opt_idx = j / 16;
                mask_opt = data_mask_opt[mo_offset + mask_opt_idx];
            }
            mask_opt_bits = (mask_opt >> ((j % 16) * 2)) & 0x3;
            if (mask_opt_bits == MASK_OPT_ALL_NEG_INF) {
                // skip this block
                continue;
            }
            // Only load if the block is not all zeros
            if (mask_opt_bits != MASK_OPT_ALL_ZERO) {
                bool nem1_bounds_check = !(p.gqa_ratio > 1) && (p.nem1 % Br) != 0;

                float max_mask = NEG_FLT_MAX_OVER_2;
                barrier();
                [[unroll]] for (uint32_t idx = 0; idx < Bc * Br; idx += gl_WorkGroupSize.x) {
                    uint32_t c = (idx + tid) % Bc;
                    uint32_t r = (idx + tid) / Bc;
                    if (idx + tid < Bc * Br) {
                        uint32_t kcol;
                        bool kv_active = fa_kv_index_c(FA_KV_CHECK, j * Bc + c, kcol);
                        if (kv_active && (!nem1_bounds_check || i * Br + r < p.nem1)) {
                            FLOAT_TYPE m = FLOAT_TYPE(data_m[m_offset + (i * Br + r) * m_stride + kcol]);
                            masksh[c * masksh_stride + r] = m;
                            max_mask = max(max_mask, float(m));
                        } else {
                            masksh[c * masksh_stride + r] = USE_SPARSE ? FLOAT_TYPE(NEG_FLT_MAX_OVER_2) : FLOAT_TYPE(0);
                        }
                    }
                }
                // skip the block if the mask is entirely -inf
                bool all_less = subgroupAll(max_mask <= NEG_FLT_MAX_OVER_2);
                barrier();
                if (gl_SubgroupInvocationID == 0) {
                    tmpsh[gl_SubgroupID] = all_less ? NEG_FLT_MAX_OVER_2 : 0.0f;
                }
                barrier();
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    max_mask = max(max_mask, tmpsh[s]);
                }
                if (max_mask <= NEG_FLT_MAX_OVER_2) {
                    continue;
                }
            }
        }

        ACC_TYPE Sf[rows_per_thread][cols_per_thread];
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                Sf[r][c] = ACC_TYPE(0.0);
            }
        }

        if (SHMEM_STAGING != 0) {
            barrier();
#ifndef MMQ
            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSK / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSK / 4);
                uint32_t c = (idx + tid) / (HSK / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSK / 4 || c < Bc) {
                    FLOAT_TYPEV4 K_Tf = FLOAT_TYPEV4(0);
                    uint32_t kcol;
                    if (fa_kv_index_c(FA_KV_CHECK, j * Bc + c, kcol)) {
                        if (USE_DECODE_K) {
                            uint coord = kcol * k_stride * BLOCK_SIZE_K + 4 * d;
                            uint ib = coord / BLOCK_SIZE_K;
                            uint iqs = (coord % BLOCK_SIZE_K);
                            K_Tf = dequantize4(ib, iqs, k_offset, BINDING_IDX_K);
                        } else {
                            K_Tf = FLOAT_TYPEV4(data_kv4[k_offset / 4 + kcol * k_stride / 4 + d]);
                        }
                    }

                    kvsh[c * kvsh_stride + d] = K_Tf;
                }
            }
#else // MMQ
            const uint ints_per_block = 8u / fa_quant_r_mmq(FaTypeK);
            const uint quant_iters = Bc * HSK / 32 * ints_per_block;
            [[unroll]] for (uint32_t idx = 0; idx < quant_iters; idx += gl_WorkGroupSize.x) {
                const uint32_t iqs = (idx + tid) % ints_per_block;
                const uint32_t ib = (idx + tid) / ints_per_block;
                const uint32_t c = ib / (HSK / 32);
                const uint32_t block = ib % (HSK / 32);
                if (idx + gl_WorkGroupSize.x <= quant_iters || c < Bc) {
                    const uint buf_ib = c * qf_stride + block;
                    uint32_t kcol;
                    if (fa_kv_index_c(FA_KV_CHECK, j * Bc + c, kcol)) {
                        const uint global_ib = kcol * k_stride + block;
                        k_block_to_shmem(buf_ib, global_ib, iqs, k_offset);
                    } else {
                        k_block_to_shmem_zero(buf_ib, iqs);
                    }
                }
            }
#endif // MMQ
            barrier();
        }

#ifndef MMQ
        // More d iterations means Q register caching becomes relevant
        // Few iterations means the additional registers needed are worse than the speed-up from caching
        if (HSK_per_thread / 4 > 4) {
            [[unroll]] for (uint32_t d = 0; d < HSK_per_thread / 4; ++d) {
                FLOAT_TYPEV4 Q_cache[rows_per_thread];
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                    Q_cache[r] = Qf[tile_row(r) * qf_stride + d * D_split + d_tid];
                }

                [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                    uint32_t kcol;
                    bool kv_active = fa_kv_index_c(FA_KV_CHECK, j * Bc + c * cols_per_iter + col_tid, kcol);
                    if (!kv_active) {
                        continue;
                    }

                    FLOAT_TYPEV4 K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else if (USE_DECODE_K) {
                        uint coord = kcol * k_stride * BLOCK_SIZE_K + 4 * (d * D_split + d_tid);
                        uint ib = coord / BLOCK_SIZE_K;
                        uint iqs = (coord % BLOCK_SIZE_K);
                        K_Tf = dequantize4(ib, iqs, k_offset, BINDING_IDX_K);
                    } else {
                        K_Tf = FLOAT_TYPEV4(data_kv4[k_offset / 4 + kcol * k_stride / 4 + d * D_split + d_tid]);
                    }
                    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                        Sf[r][c] = dot_product(Q_cache[r], K_Tf, Sf[r][c]);
                    }
                }
            }
        } else {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                uint32_t kcol;
                bool kv_active = fa_kv_index_c(FA_KV_CHECK, j * Bc + c * cols_per_iter + col_tid, kcol);
                if (!kv_active) {
                    continue;
                }

                [[unroll]] for (uint32_t d = 0; d < HSK_per_thread / 4; ++d) {
                    FLOAT_TYPEV4 K_Tf;
                    if (SHMEM_STAGING != 0) {
                        K_Tf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                    } else if (USE_DECODE_K) {
                        uint coord = kcol * k_stride * BLOCK_SIZE_K + 4 * (d * D_split + d_tid);
                        uint ib = coord / BLOCK_SIZE_K;
                        uint iqs = (coord % BLOCK_SIZE_K);
                        K_Tf = dequantize4(ib, iqs, k_offset, BINDING_IDX_K);
                    } else {
                        K_Tf = FLOAT_TYPEV4(data_kv4[k_offset / 4 + kcol * k_stride / 4 + d * D_split + d_tid]);
                    }
                    [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                        Sf[r][c] = dot_product(Qf[tile_row(r) * qf_stride + d * D_split + d_tid], K_Tf, Sf[r][c]);
                    }
                }
            }
        }
#else // MMQ
        const uint hsk4 = HSK_per_thread / 4;
        const uint d_per_step = (hsk4 % 8 == 0) ? 8 :
                                (hsk4 % 4 == 0) ? 4 :
                                (hsk4 % 2 == 0) ? 2 : 1;

        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
            uint32_t kcol;
            if (!fa_kv_index_c(FA_KV_CHECK, j * Bc + c * cols_per_iter + col_tid, kcol)) {
                continue;
            }

            [[unroll]] for (uint32_t d_block = 0; d_block < HSK_per_thread / 4; d_block += d_per_step) {
                int32_t k_quants[d_per_step];
                ACC_TYPEV2 k_dm;

                // Q4_*/Q5_* take the block-8 fast path when one step covers a full
                // block; Q8_0 always goes through the per-int get_k_qs* helpers
                // (its qs is byte-packed, not nibble-packed).
                const bool block8_fast = (d_per_step == 8) && (FaTypeK != GGML_TYPE_Q8_0);

                if (SHMEM_STAGING != 0) {
                    const uint k_block_idx = (d_tid * (HSK_per_thread / 4) + d_block) / 8;
                    const uint buf_ib = (c * cols_per_iter + col_tid) * qf_stride + k_block_idx;
                    k_dm = ACC_TYPEV2(kblocksh[buf_ib].dm);

                    if (block8_fast) {
                        const bool has_qh = (FaTypeK == GGML_TYPE_Q5_0) || (FaTypeK == GGML_TYPE_Q5_1);
                        [[unroll]] for (uint32_t d = 0; d < 4; d++) {
                            uint vui = kblocksh[buf_ib].qs[d];
                            k_quants[d    ] = int32_t( vui       & 0x0F0F0F0F);
                            k_quants[d + 4] = int32_t((vui >> 4) & 0x0F0F0F0F);
                            if (has_qh) {
                                uint qh_lo = (kblocksh[buf_ib].qh >> (d * 4)) & 0xF;
                                uint qh_hi = (kblocksh[buf_ib].qh >> (d * 4 + 16)) & 0xF;
                                k_quants[d    ] |= int32_t((qh_lo * 0x02040810u) & 0x10101010u);
                                k_quants[d + 4] |= int32_t((qh_hi * 0x02040810u) & 0x10101010u);
                            }
                        }
                    } else {
                        [[unroll]] for (uint32_t d = 0; d < d_per_step; d++) {
                            k_quants[d] = get_k_qs_shmem(buf_ib, (d_tid * (HSK_per_thread / 4) + d_block) % 8 + d);
                        }
                    }
                } else {
                    const uint coord = kcol * k_stride * BLOCK_SIZE_K + 4 * (d_tid * (HSK_per_thread / 4) + d_block);
                    const uint ib = coord / BLOCK_SIZE_K;
                    const uint iqs = (coord % BLOCK_SIZE_K);

                    k_dm = ACC_TYPEV2(get_k_scale(ib, k_offset));

                    if (block8_fast) {
                        fa_k_qs_block8 blk = get_k_qs_block8(ib, k_offset);
                        [[unroll]] for (uint32_t d = 0; d < 8; d++) {
                            k_quants[d] = blk.qs[d];
                        }
                    } else {
                        [[unroll]] for (uint32_t d = 0; d < d_per_step; d++) {
                            k_quants[d] = get_k_qs(ib, iqs + d * 4, k_offset);
                        }
                    }
                }

                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                    const uint qib = tile_row(r) * qf_stride + (d_tid * (HSK_per_thread / 4) + d_block) / 8;
                    const uint qiqs = (d_tid * (HSK_per_thread / 4) + d_block) % 8;

                    int32_t acc = 0;
                    [[unroll]] for (uint32_t d = 0; d < d_per_step; d++) {
                        acc += dotPacked4x8EXT(Qf[qib].qs[qiqs + d], k_quants[d]);
                    }

                    Sf[r][c] += ACC_TYPE(acc) * ACC_TYPE(Qf[qib].ds.x) * k_dm.x;
                    if ((d_tid * (HSK_per_thread / 4) + d_block) % 8 == 0) {
                        Sf[r][c] += k_dot_correction(qib, k_dm);
                    }
                }
            }
        }
#endif // MMQ

        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
            // Compute sum across the D_split
            [[unroll]] for (uint s = D_split / 2; s > 0; s >>= 1) {
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                    Sf[r][c] += subgroupShuffleXor(Sf[r][c], s);
                }
            }
        }

        if (LOGIT_SOFTCAP) {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                    Sf[r][c] = ACC_TYPE(p.logit_softcap * tanh(Sf[r][c]));
                }
            }
        }

        if (MASK_ENABLE && mask_opt_bits != MASK_OPT_ALL_ZERO) {
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                    FLOAT_TYPE mvf = masksh[(c * cols_per_iter + col_tid) * masksh_stride + tile_row(r)];

                    Sf[r][c] += slope[r]*mvf;
                }
            }
        }

        float eMf[rows_per_thread];
        [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
            float rowmaxf = NEG_FLT_MAX_OVER_2;
            [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
                if (FA_KV_CHECK && j * Bc + c * cols_per_iter + col_tid >= KV) {
                    continue;
                }
                rowmaxf = max(rowmaxf, float(Sf[r][c]));
            }
            rowmaxf += FATTN_KQ_MAX_OFFSET;
            float Moldf = Mf[r];

            // M = max(rowmax, Mold)
            // P = e^(S - M)
            // eM = e^(Mold - M)
            Mf[r] = max(rowmaxf, Moldf);
            eMf[r] = exp(Moldf - Mf[r]);
            Lf[r] = eMf[r]*Lf[r];
        }

        [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++d) {
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                Of[r][d] = FLOAT_TYPE(eMf[r]) * Of[r][d];
            }
        }

        if (SHMEM_STAGING != 0) {
            barrier();
            [[unroll]] for (uint32_t idx = 0; idx < Bc * HSV / 4; idx += gl_WorkGroupSize.x) {
                uint32_t d = (idx + tid) % (HSV / 4);
                uint32_t c = (idx + tid) / (HSV / 4);
                if (idx + gl_WorkGroupSize.x <= Bc * HSV / 4 || c < Bc) {
                    FLOAT_TYPEV4 V_Tf = FLOAT_TYPEV4(0);
                    uint32_t vcol;
                    if (fa_kv_index_c(FA_KV_CHECK, j * Bc + c, vcol)) {
                        if (USE_DECODE_V) {
                            uint coord = vcol * v_stride * BLOCK_SIZE_V + 4 * d;
                            uint ib = coord / BLOCK_SIZE_V;
                            uint iqs = (coord % BLOCK_SIZE_V);
                            V_Tf = dequantize4(ib, iqs, v_offset, BINDING_IDX_V);
                        } else {
                            V_Tf = FLOAT_TYPEV4(data_vv4[v_offset / 4 + vcol * v_stride / 4 + d]);
                        }
                    }

                    kvsh[c * kvsh_stride + d] = V_Tf;
                }
            }
            barrier();
        }

        [[unroll]] for (uint32_t c = 0; c < cols_per_thread; ++c) {
            uint32_t vcol;
            bool kv_active = fa_kv_index_c(FA_KV_CHECK, j * Bc + c * cols_per_iter + col_tid, vcol);
            if (!kv_active) {
                continue;
            }

            FLOAT_TYPE Pf[rows_per_thread];
            [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                Pf[r] = FLOAT_TYPE(exp(float(Sf[r][c]) - Mf[r]));
                Lf[r] += Pf[r];
            }

            [[unroll]] for (uint32_t d = 0; d < HSV_per_thread / 4; ++d) {
                FLOAT_TYPEV4 Vf;
                if (SHMEM_STAGING != 0) {
                    Vf = kvsh[(c * cols_per_iter + col_tid) * kvsh_stride + (d * D_split + d_tid)];
                } else if (USE_DECODE_V) {
                    uint coord = vcol * v_stride * BLOCK_SIZE_V + 4 * (d * D_split + d_tid);
                    uint ib = coord / BLOCK_SIZE_V;
                    uint iqs = (coord % BLOCK_SIZE_V);
                    Vf = dequantize4(ib, iqs, v_offset, BINDING_IDX_V);
                } else {
                    Vf = FLOAT_TYPEV4(data_vv4[v_offset / 4 + vcol * v_stride / 4 + d * D_split + d_tid]);
                }
                [[unroll]] for (uint32_t r = 0; r < rows_per_thread; ++r) {
                    Of[r][d] += FLOAT_TYPEV4(Pf[r] * Vf);
                }
            }
        }
