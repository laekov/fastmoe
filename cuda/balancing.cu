#include <cstdio>
#include "balancing.cuh"
#include "global_exchange.h"
#include <torch/extension.h>

/* 
 * note that due to limit of cuda atomic operator, capacity should be int32
 */
torch::Tensor _limit_by_capacity(
        torch::Tensor expert_count, torch::Tensor capacity,
        long n_expert, long n_worker) {
    CHECK_INPUT(expert_count);
    CHECK_INPUT(capacity);
    auto expert_count_ack = torch::empty_like(expert_count);
    auto smgr = getCudaStreamManager(expert_count.device().index());
    fmoe_cuda_limit_by_capacity_impl(
            expert_count.data_ptr<long>(),
            capacity.data_ptr<int>(),
            expert_count_ack.data_ptr<long>(),
            n_expert, n_worker, smgr);
    return expert_count_ack;
}

torch::Tensor _prune_gate_by_capacity(
        torch::Tensor gate_idx, torch::Tensor expert_count,
        long n_expert, long n_worker) {
    auto smgr = getCudaStreamManager(expert_count.device().index());
    auto batch_size = gate_idx.numel();
    auto opt = torch::TensorOptions()
        .dtype(gate_idx.dtype())
        .device(gate_idx.device());
    auto new_gate_idx = torch::empty(gate_idx.sizes(), opt);
    fmoe_cuda_prune_gate_by_capacity_impl(
            gate_idx.data_ptr<long>(),
            new_gate_idx.data_ptr<long>(),
            expert_count.data_ptr<int>(),
            batch_size, n_expert, n_worker, smgr);
    return new_gate_idx;
}

template<class T>
T* _cudamalloc(size_t sz) {
    T* dptr;
    cudaMalloc(&dptr, sz * sizeof(T));
    return dptr;
}

template<class T>
T* _h2d(const T* hptr, T* dptr, size_t sz) {
    cudaMemcpy(dptr, hptr, sz * sizeof(T), cudaMemcpyHostToDevice);
    return dptr;
}
template<class T>
T* _h2d(T* hptr, size_t sz) {
    T* dptr = _cudamalloc<T>(sz);
    return _h2d(hptr, dptr, sz);
}
template<class T>
T* _d2h(const T* dptr, T* hptr, size_t sz) {
    cudaMemcpy(hptr, dptr, sz * sizeof(T), cudaMemcpyDeviceToHost);
    return hptr;
}
template<class T>
T* _d2h(const T* dptr, size_t sz) {
    T* hptr = new T[sz];
    return _d2h(dptr, hptr, sz);
}

#ifdef FMOE_USE_NCCL

#include <nccl.h>

#define UPDATE_COUNTERS(__count__) { \
    if (i == rank) { \
        lec[j] += (__count__); \
    } \
    if (j == rank) { \
        gec[i] += (__count__); \
        cap -= (__count__); \
    } \
}

std::vector<torch::Tensor> _swipe_once(
        torch::Tensor gate_idx, torch::Tensor capacity,
        long n_expert, long n_worker, long bias) {
    auto device_idx = gate_idx.device().index();
    auto smgr = getCudaStreamManager(device_idx);
    int rank;
    ncclCommUserRank(smgr->ncclcomm, &rank);
    cudaSetDevice(device_idx);

    auto capacity_new = capacity.clone();
    auto cap = capacity_new.item<long>();

    long batch_size = gate_idx.size(0);
    auto gate_idx_cpu = gate_idx.cpu();
    long* gidx = gate_idx_cpu.data_ptr<long>();

    /* Local count and exchange */
    long *lec = new long[n_worker];
    memset(lec, 0, n_worker * sizeof(long));
    for (long i = 0; i < batch_size; ++i) {
        ++lec[gidx[i] / n_expert];
    }
    long *d_lec = _h2d(lec, n_worker), *d_gec = _cudamalloc<long>(n_worker);
    fmoe_cuda_expert_exchange_impl(d_lec, d_gec, 1, n_worker, smgr);
    long *gec = _d2h(d_gec, n_worker);

    /* Limit number of incoming samples */
    long *drop_count = new long[n_worker];
    memset(drop_count, 0, n_worker * sizeof(long));
    for (long i = 0; i < n_worker; ++i) {
        if (cap >= gec[i]) {
            drop_count[i] = 0;
            cap -= gec[i];
        } else {
            drop_count[i] = gec[i] - cap;
            gec[i] = cap;
            cap = 0;
        }
    }

    /* Send limit information back */
    _h2d(gec, d_gec, n_worker);
    fmoe_cuda_expert_exchange_impl(d_gec, d_lec, 1, n_worker, smgr);
    _d2h(d_lec, lec, n_worker);

    auto d_dropcount = _h2d(drop_count, n_worker);
    ncclAllReduce(d_dropcount, d_dropcount, n_worker, ncclInt64, ncclSum,
            smgr->ncclcomm, smgr->stream());
    _d2h(d_dropcount, drop_count, n_worker);

    auto d_gcap = _cudamalloc<long>(n_worker);
    _h2d(&cap, d_gcap + rank, 1);
    ncclAllGather(d_gcap + rank, d_gcap, 1, ncclInt64,
            smgr->ncclcomm, smgr->stream());
    auto gcap = _d2h(d_gcap, n_worker);

    /* Re-assign and update counters */
    for (long i = 0, j = 0; i < n_worker; ++i) {
        while (drop_count[i] > 0) {
            if (drop_count[i] > gcap[j]) {
                drop_count[i] -= gcap[j];
                UPDATE_COUNTERS(gcap[j]);
                ++j;
            } else {
                gcap[j] -= drop_count[i];
                UPDATE_COUNTERS(drop_count[i]);
                break;
            }
        }
    }
    for (long i = 0; i < batch_size; ++i) {
        auto widx = gidx[i] / n_expert;
        if (lec[widx] > 0) {
            --lec[widx];
        } else {
            gidx[i] = -1;
        }
    }
    for (long i = 0, k = 0; i < batch_size; ++i) {
        if (gidx[i] != -1) {
            continue;
        }
        for (; lec[k] == 0; ++k);
        --lec[k];
        gidx[i] = k * n_expert + bias;
    }
    *capacity_new.data_ptr<long>() = cap;

    delete [] drop_count;
    delete [] lec;
    delete [] gec;
    delete [] gcap;

    cudaFree(d_dropcount);
    cudaFree(d_lec);
    cudaFree(d_gec);
    cudaFree(d_gcap);

    return {gate_idx_cpu, capacity_new};
}

#undef UPDATE_COUNTERS

#endif
