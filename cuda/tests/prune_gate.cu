#include "../balancing.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* args[]) {
    int n_worker = atoi(args[1]);
    int n_expert = atoi(args[2]);
    int batch_size = atoi(args[3]);
    int tot_expert = n_worker * n_expert;

    long* gate_idx = new long[batch_size];
    long* n_gate_idx = new long[batch_size];

    long* lec = new long[tot_expert];
    memset(lec, 0, sizeof(long) * tot_expert);

    for (int i = 0; i < batch_size; ++i) {
        gate_idx[i] = rand() % tot_expert;
        ++lec[gate_idx[i]];
    }
    for (int i = 0; i < tot_expert; ++i) {
        lec[i] >>= 1;
    }
    long* g_lec;
    cudaMalloc(&g_lec, sizeof(long) * tot_expert);
    cudaMemcpy(g_lec, lec, sizeof(long) * tot_expert, cudaMemcpyHostToDevice);

    int* g_new_lec;
    cudaMalloc(&g_new_lec, sizeof(int) * tot_expert);

    long* g_gate_idx;
    cudaMalloc(&g_gate_idx, sizeof(long) * batch_size);
    cudaMemcpy(g_gate_idx, gate_idx, sizeof(long) * batch_size, cudaMemcpyHostToDevice);

    auto smgr = getCudaStreamManager(0);
    fmoe_cuda_prune_gate_by_capacity_impl(g_gate_idx, g_lec, g_new_lec,
            batch_size, n_expert, n_worker, smgr);
    cudaMemcpy(n_gate_idx, g_gate_idx, sizeof(long) * batch_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        printf("%ld %ld (%d)\n", gate_idx[i], n_gate_idx[i], lec[gate_idx[i]]);
    }
}
