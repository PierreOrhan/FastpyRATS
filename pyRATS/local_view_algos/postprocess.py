import numpy as np
import multiprocess as mp
from multiprocess import shared_memory
import copy
from scipy.spatial.distance import pdist, squareform

from ..common_ import *
from ..util_ import compute_zeta,Param, fast_compute_zeta
from ..nbrhd_graph_ import NbrhdGraph
import torch
def postprocess(nbrhd_graph: NbrhdGraph, local_param_pre: Param, opts):
    U = nbrhd_graph.neigh_ind
    X = local_param_pre.X
    n = U.shape[0]
    local_param_eval = local_param_pre.eval_({'view_index': torch.arange(n,device="cuda"),
                                              'data_index': U})
    
    local_param_pre.compute_local_distortion_(nbrhd_graph, local_param_eval)    

    print(r'Maximum local distortion before postprocessing is: %0.4f' % torch.max(local_param_pre.zeta))

    new_param_of = torch.arange(n, dtype=int,device="cuda") # new param of

    ### PostProcessing:
    some_param_changed = True
    while some_param_changed:
        param_changed_old = None
        if param_changed_old is None:
            cand_k = U
        else:
            ## Assuming param_changed_old and U have unique elements:
            # We can use the following to take the intersection in torch in a batched way
            tmp_paramUu, counts = torch.cat([param_changed_old, U],dim=1).unique(dim=1,return_counts=True)
            # The intersection can be of different length, so we have to first expand here:
            cand_k = [tmp_paramUu[e,torch.where(counts[e,:].gt(1))] for e in range(tmp_paramUu.shape[0])]
            # and then Pad:
            ## Pad the candidate_k list to make it a rectangular tensor
            # We pad with the index itself, such that if we compute zeta_{kk} it is the same as before
            max_len = max([len(c) for c in cand_k])
            cand_k = torch.stack([torch.nn.functional.pad(c,(0, max_len - c.shape[0]),value=idc) 
                                    for idc,c in enumerate(cand_k)])
            # TODO check: cand_k should be of shape (n, max_len)

        ## For every points, and for every candidate, 
        # compute the local distortion if we replace the parameter:
        view_index = new_param_of[cand_k].reshape(-1)
        U_stack = torch.stack([U for _ in range(cand_k.shape[1])],dim=1).reshape(-1, U.shape[1])
        local_param_eval_kponUk = local_param_pre.eval_({'view_index': view_index,
                                                        'data_index': U_stack})
        # (n*max_len, k, d)
        zeta_kkp = fast_compute_zeta(nbrhd_graph, local_param_eval_kponUk, X,U_stack)
        zeta_kkp = zeta_kkp.reshape(n, -1) #(n, max_len)
        
        # For every point where there is at least one candidate with lower distortion, we pick the best one
        improved = zeta_kkp < local_param_pre.zeta[:,None]
        improved_any = improved.any(dim=1)
        improved_idx = torch.where(improved_any)[0]
        if len(improved_idx)>0:
            best_cand_idx = torch.argmin(zeta_kkp, dim=1)[improved_any]
            # Update the parameters for the improved ones
            local_param_pre.zeta[improved_idx] = zeta_kkp[improved_idx,best_cand_idx]
            new_param_of[improved_idx] = new_param_of[cand_k[improved_idx,best_cand_idx]]
            some_param_changed = True
        else:
            some_param_changed = False
        # print(torch.sum(improved_any))

    local_param_pre.replace_(new_param_of)
    print(f'Maximum local distoriton after post-processing is: %0.4f' % (torch.max(local_param_pre.zeta)))
    return local_param_pre

    # if len(cand_k)==0:
    #     param_changed_buf[k] = False
    #     continue
    # d_e_k = d_e[np.ix_(U_k,U_k)]
    
    # for kp in cand_k:
    #     Psi_kp_on_U_k = local_param.eval_({'view_index': new_param_of[kp],
    #                                         'data_mask': U_k})
    #     zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)
    #     # if zeta_{kk'} < zeta_{kk}
    #     if zeta_kkp < zeta_[k]:
    #         zeta_[k] = zeta_kkp
    #         new_param_of_[k-start_ind] = new_param_of[kp]
    #         param_changed_for_k = True
    # param_changed_buf[k] = param_changed_for_k

    
    # local_param = copy.deepcopy(local_param_pre)
    # zeta = np.zeros(n)
    # for k in range(n):
    #     U_k = U[k,:].indices
    #     d_e_k = d_e[np.ix_(U_k,U_k)]
    #     zeta[k] = compute_zeta(d_e_k, local_param.eval_({'view_index': k, 'data_mask': U_k}))
    # local_param.zeta = zeta
    
    # n_proc = opts['n_proc']
    # barrier = mp.Barrier(n_proc)
    # pcb = np.zeros(n, dtype=bool) # param changed buffer and converge flag
    # npo = np.arange(n, dtype=int) # new param of
    # pcb_dtype = pcb.dtype
    # pcb_shape = pcb.shape
    # npo_dtype = npo.dtype
    # npo_shape = npo.shape
    # zeta_shape = zeta.shape
    # zeta_dtype = zeta.dtype

    # shm_pcb = shared_memory.SharedMemory(create=True, size=pcb.nbytes)
    # np_pcb = np.ndarray(pcb_shape, dtype=pcb_dtype, buffer=shm_pcb.buf)
    # np_pcb[:] = pcb[:]
    # shm_npo = shared_memory.SharedMemory(create=True, size=npo.nbytes)
    # np_npo = np.ndarray(npo_shape, dtype=npo_dtype, buffer=shm_npo.buf)
    # np_npo[:] = npo[:]
    # shm_zeta = shared_memory.SharedMemory(create=True, size=zeta.nbytes)
    # np_zeta = np.ndarray(zeta_shape, dtype=zeta_dtype, buffer=shm_zeta.buf)
    # np_zeta[:] = zeta[:]

    # shm_pcb_name = shm_pcb.name
    # shm_npo_name = shm_npo.name
    # shm_zeta_name = shm_zeta.name

    # def target_proc(p_num, chunk_sz, barrier, U, local_param, d_e):
    #     existing_shm_pcb = shared_memory.SharedMemory(name=shm_pcb_name)
    #     param_changed_buf = np.ndarray(pcb_shape, dtype=pcb_dtype,
    #                                     buffer=existing_shm_pcb.buf)
    #     existing_shm_npo = shared_memory.SharedMemory(name=shm_npo_name)
    #     new_param_of = np.ndarray(npo_shape, dtype=npo_dtype,
    #                                 buffer=existing_shm_npo.buf)
    #     existing_shm_zeta = shared_memory.SharedMemory(name=shm_zeta_name)
    #     zeta_ = np.ndarray(zeta_shape, dtype=zeta_dtype,
    #                                 buffer=existing_shm_zeta.buf)

    #     start_ind = p_num*chunk_sz
    #     if p_num == (n_proc-1):
    #         end_ind = n
    #     else:
    #         end_ind = (p_num+1)*chunk_sz

    #     param_changed_old = None
    #     new_param_of_ = np.arange(start_ind, end_ind)
    #     N_replaced = n
    #     while N_replaced: # while not converged
    #         for k in range(start_ind, end_ind):
    #             param_changed_for_k = False
    #             U_k = U[k,:].indices
    #             if param_changed_old is None:
    #                 cand_k = U_k.tolist()
    #             else:
    #                 cand_k = list(param_changed_old.intersection(U_k.tolist()))
    #             if len(cand_k)==0:
    #                 param_changed_buf[k] = False
    #                 continue
    #             d_e_k = d_e[np.ix_(U_k,U_k)]
                
    #             for kp in cand_k:
    #                 Psi_kp_on_U_k = local_param.eval_({'view_index': new_param_of[kp],
    #                                                     'data_mask': U_k})
    #                 zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)
    #                 # if zeta_{kk'} < zeta_{kk}
    #                 if zeta_kkp < zeta_[k]:
    #                     zeta_[k] = zeta_kkp
    #                     new_param_of_[k-start_ind] = new_param_of[kp]
    #                     param_changed_for_k = True
    #             param_changed_buf[k] = param_changed_for_k
            
    #         barrier.wait()
    #         new_param_of[start_ind:end_ind] = new_param_of_
    #         param_changed_old = set(np.where(param_changed_buf)[0])
    #         N_replaced = len(param_changed_old)
    #         barrier.wait()
    #         if p_num == 0:
    #             print("#Param replaced: %d, max distortion: %f" % (N_replaced, np.max(zeta_)))
                    
    #     existing_shm_pcb.close()
    #     existing_shm_npo.close()
    #     existing_shm_zeta.close()

    # proc = []
    # chunk_sz = int(n/n_proc)
    # for p_num in range(n_proc):
    #     proc.append(mp.Process(target=target_proc, args=(p_num,chunk_sz, barrier,
    #                                                      U, local_param, d_e),
    #                             daemon=True))
    #     proc[-1].start()

    # for p_num in range(n_proc):
    #     proc[p_num].join()

    # npo[:] = np_npo[:]
    # local_param.zeta[:] = np_zeta[:]

    # del np_npo
    # shm_npo.close()
    # shm_npo.unlink()
    # del np_zeta
    # shm_zeta.close()
    # shm_zeta.unlink()
    # del np_pcb
    # shm_pcb.close()
    # shm_pcb.unlink()

    # local_param.replace_(npo)
    # print(f'Maximum local distoriton after post-processing is: %0.4f' % (np.max(local_param.zeta)))
    # return local_param


# Old version which computes pairwise local distances on the run
# def postprocess(X, nbrhd_graph, local_param_pre, opts):
#     n = nbrhd_graph.get_num_nodes()
#     local_param = copy.deepcopy(local_param_pre)

#     n_proc = opts['n_proc']
#     barrier = mp.Barrier(n_proc)
#     pcb = np.zeros(n, dtype=bool) # param changed buffer and converge flag
#     npo = np.arange(n, dtype=int) # new param of

#     zeta = np.zeros(n)
#     for k in range(n):
#         U_k = nbrhd_graph.get_nbr_inds(k)
#         d_e_k = squareform(pdist(X[U_k,:], metric=opts['metric']))
#         zeta[k] = compute_zeta(d_e_k, local_param.eval_({'view_index': k, 'data_mask': U_k}))
#     local_param.zeta = zeta
#     print(r'Maximum local distoriton before postptocessing is: %0.4f' % np.max(zeta))

#     pcb_dtype = pcb.dtype
#     pcb_shape = pcb.shape
#     npo_dtype = npo.dtype
#     npo_shape = npo.shape
#     zeta_shape = zeta.shape
#     zeta_dtype = zeta.dtype

#     shm_pcb = shared_memory.SharedMemory(create=True, size=pcb.nbytes)
#     np_pcb = np.ndarray(pcb_shape, dtype=pcb_dtype, buffer=shm_pcb.buf)
#     np_pcb[:] = pcb[:]
#     shm_npo = shared_memory.SharedMemory(create=True, size=npo.nbytes)
#     np_npo = np.ndarray(npo_shape, dtype=npo_dtype, buffer=shm_npo.buf)
#     np_npo[:] = npo[:]
#     shm_zeta = shared_memory.SharedMemory(create=True, size=zeta.nbytes)
#     np_zeta = np.ndarray(zeta_shape, dtype=zeta_dtype, buffer=shm_zeta.buf)
#     np_zeta[:] = zeta[:]

#     shm_pcb_name = shm_pcb.name
#     shm_npo_name = shm_npo.name
#     shm_zeta_name = shm_zeta.name

#     def target_proc(p_num, chunk_sz, barrier, nbrhd_graph, local_param, X):
#         existing_shm_pcb = shared_memory.SharedMemory(name=shm_pcb_name)
#         param_changed_buf = np.ndarray(pcb_shape, dtype=pcb_dtype,
#                                         buffer=existing_shm_pcb.buf)
#         existing_shm_npo = shared_memory.SharedMemory(name=shm_npo_name)
#         new_param_of = np.ndarray(npo_shape, dtype=npo_dtype,
#                                     buffer=existing_shm_npo.buf)
#         existing_shm_zeta = shared_memory.SharedMemory(name=shm_zeta_name)
#         zeta_ = np.ndarray(zeta_shape, dtype=zeta_dtype,
#                                     buffer=existing_shm_zeta.buf)

#         start_ind = p_num*chunk_sz
#         if p_num == (n_proc-1):
#             end_ind = n
#         else:
#             end_ind = (p_num+1)*chunk_sz

#         param_changed_old = None
#         new_param_of_ = np.arange(start_ind, end_ind)
#         N_replaced = n
#         while N_replaced: # while not converged
#             for k in range(start_ind, end_ind):
#                 param_changed_for_k = False
#                 U_k = nbrhd_graph.get_nbr_inds(k)
#                 if param_changed_old is None:
#                     cand_k = U_k.tolist()
#                 else:
#                     cand_k = list(param_changed_old.intersection(U_k.tolist()))
#                 if len(cand_k)==0:
#                     param_changed_buf[k] = False
#                     continue
#                 d_e_k = squareform(pdist(X[U_k,:], metric=opts['metric']))
                
#                 for kp in cand_k:
#                     Psi_kp_on_U_k = local_param.eval_({'view_index': new_param_of[kp],
#                                                         'data_mask': U_k})
#                     zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)
#                     # if zeta_{kk'} < zeta_{kk}
#                     if zeta_kkp < zeta_[k]:
#                         zeta_[k] = zeta_kkp
#                         new_param_of_[k-start_ind] = new_param_of[kp]
#                         param_changed_for_k = True
#                 param_changed_buf[k] = param_changed_for_k
            
#             barrier.wait()
#             new_param_of[start_ind:end_ind] = new_param_of_
#             param_changed_old = set(np.where(param_changed_buf)[0])
#             N_replaced = len(param_changed_old)
#             barrier.wait()
#             if p_num == 0:
#                 print("#Param replaced: %d, max distortion: %f" % (N_replaced, np.max(zeta_)))
                    
#         existing_shm_pcb.close()
#         existing_shm_npo.close()
#         existing_shm_zeta.close()

#     proc = []
#     chunk_sz = int(n/n_proc)
#     for p_num in range(n_proc):
#         proc.append(mp.Process(target=target_proc, args=(p_num,chunk_sz, barrier,
#                                                             nbrhd_graph, local_param, X),
#                                 daemon=True))
#         proc[-1].start()

#     for p_num in range(n_proc):
#         proc[p_num].join()

#     npo[:] = np_npo[:]
#     local_param.zeta[:] = np_zeta[:]

#     del np_npo
#     shm_npo.close()
#     shm_npo.unlink()
#     del np_zeta
#     shm_zeta.close()
#     shm_zeta.unlink()
#     del np_pcb
#     shm_pcb.close()
#     shm_pcb.unlink()

#     local_param.replace_(npo)
#     return local_param