import numpy as np
import multiprocess as mp
from multiprocess import shared_memory
import copy
from scipy.spatial.distance import pdist, squareform

from ..common_ import *
from ..util_ import compute_zeta

def postprocess(U, d_e, local_param_pre, opts):
    n = U.shape[0]
    local_param = copy.deepcopy(local_param_pre)

    n_proc = opts['n_proc']
    barrier = mp.Barrier(n_proc)
    pcb = np.zeros(n, dtype=bool) # param changed buffer and converge flag
    npo = np.arange(n, dtype=int) # new param of

    zeta = np.zeros(n)
    for k in range(n):
        U_k = U[k,:].indices
        d_e_k = d_e[np.ix_(U_k,U_k)]
        zeta[k] = compute_zeta(d_e_k, local_param.eval_({'view_index': k, 'data_mask': U_k}))
    local_param.zeta = zeta
    print(r'Maximum local distoriton before postptocessing is: %0.4f' % np.max(zeta))

    pcb_dtype = pcb.dtype
    pcb_shape = pcb.shape
    npo_dtype = npo.dtype
    npo_shape = npo.shape
    zeta_shape = zeta.shape
    zeta_dtype = zeta.dtype

    shm_pcb = shared_memory.SharedMemory(create=True, size=pcb.nbytes)
    np_pcb = np.ndarray(pcb_shape, dtype=pcb_dtype, buffer=shm_pcb.buf)
    np_pcb[:] = pcb[:]
    shm_npo = shared_memory.SharedMemory(create=True, size=npo.nbytes)
    np_npo = np.ndarray(npo_shape, dtype=npo_dtype, buffer=shm_npo.buf)
    np_npo[:] = npo[:]
    shm_zeta = shared_memory.SharedMemory(create=True, size=zeta.nbytes)
    np_zeta = np.ndarray(zeta_shape, dtype=zeta_dtype, buffer=shm_zeta.buf)
    np_zeta[:] = zeta[:]

    shm_pcb_name = shm_pcb.name
    shm_npo_name = shm_npo.name
    shm_zeta_name = shm_zeta.name

    def target_proc(p_num, chunk_sz, barrier, U, local_param, d_e):
        existing_shm_pcb = shared_memory.SharedMemory(name=shm_pcb_name)
        param_changed_buf = np.ndarray(pcb_shape, dtype=pcb_dtype,
                                        buffer=existing_shm_pcb.buf)
        existing_shm_npo = shared_memory.SharedMemory(name=shm_npo_name)
        new_param_of = np.ndarray(npo_shape, dtype=npo_dtype,
                                    buffer=existing_shm_npo.buf)
        existing_shm_zeta = shared_memory.SharedMemory(name=shm_zeta_name)
        zeta_ = np.ndarray(zeta_shape, dtype=zeta_dtype,
                                    buffer=existing_shm_zeta.buf)

        start_ind = p_num*chunk_sz
        if p_num == (n_proc-1):
            end_ind = n
        else:
            end_ind = (p_num+1)*chunk_sz

        param_changed_old = None
        new_param_of_ = np.arange(start_ind, end_ind)
        N_replaced = n
        while N_replaced: # while not converged
            for k in range(start_ind, end_ind):
                param_changed_for_k = False
                U_k = U[k,:].indices
                if param_changed_old is None:
                    cand_k = U_k.tolist()
                else:
                    cand_k = list(param_changed_old.intersection(U_k.tolist()))
                if len(cand_k)==0:
                    param_changed_buf[k] = False
                    continue
                d_e_k = d_e[np.ix_(U_k,U_k)]
                
                for kp in cand_k:
                    Psi_kp_on_U_k = local_param.eval_({'view_index': new_param_of[kp],
                                                        'data_mask': U_k})
                    zeta_kkp = compute_zeta(d_e_k, Psi_kp_on_U_k)
                    # if zeta_{kk'} < zeta_{kk}
                    if zeta_kkp < zeta_[k]:
                        zeta_[k] = zeta_kkp
                        new_param_of_[k-start_ind] = new_param_of[kp]
                        param_changed_for_k = True
                param_changed_buf[k] = param_changed_for_k
            
            barrier.wait()
            new_param_of[start_ind:end_ind] = new_param_of_
            param_changed_old = set(np.where(param_changed_buf)[0])
            N_replaced = len(param_changed_old)
            barrier.wait()
            if p_num == 0:
                print("#Param replaced: %d, max distortion: %f" % (N_replaced, np.max(zeta_)))
                    
        existing_shm_pcb.close()
        existing_shm_npo.close()
        existing_shm_zeta.close()

    proc = []
    chunk_sz = int(n/n_proc)
    for p_num in range(n_proc):
        proc.append(mp.Process(target=target_proc, args=(p_num,chunk_sz, barrier,
                                                         U, local_param, d_e),
                                daemon=True))
        proc[-1].start()

    for p_num in range(n_proc):
        proc[p_num].join()

    npo[:] = np_npo[:]
    local_param.zeta[:] = np_zeta[:]

    del np_npo
    shm_npo.close()
    shm_npo.unlink()
    del np_zeta
    shm_zeta.close()
    shm_zeta.unlink()
    del np_pcb
    shm_pcb.close()
    shm_pcb.unlink()

    local_param.replace_(npo)
    print(f'Maximum local distoriton after post-processing is: %0.4f' % (np.max(local_param.zeta)))
    return local_param


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