#include "../local_exchange.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* args[]) {
    int batch_size = atoi(args[1]);
    int n_expert = atoi(args[2]);

    long* gate_idx = new long[batch_size];
    long* n_gate_idx = new long[batch_size];
    int* ref_lec = new int[n_expert];
    memset(ref_lec, 0, sizeof(int) * n_expert);

    for (int i = 0; i < batch_size; ++i) {
        gate_idx[i] = rand() % (n_expert + 1) - 1;
        if (gate_idx[i] != -1) {
            ref_lec[gate_idx[i]] += 1;
        }
    }

    puts("ref lec");
    for (int i = 0; i < n_expert; ++i) {
        printf("%d ", ref_lec[i]);
    }
    putchar(10);

    int* g_lec;
    cudaMalloc(&g_lec, sizeof(int) * n_expert);
    cudaMemset(g_lec, 0, sizeof(int) * n_expert);
    long* g_gate_idx;
    cudaMalloc(&g_gate_idx, sizeof(long) * batch_size);
    cudaMemcpy(g_gate_idx, gate_idx, sizeof(long) * batch_size,
            cudaMemcpyHostToDevice);

    auto smgr = getCudaStreamManager(0);
    fmoe_cuda_expert_count_impl(g_gate_idx, g_lec, batch_size, n_expert, smgr);

    int* lec = new int[n_expert];
    cudaMemcpy(lec, g_lec, sizeof(int) * n_expert, cudaMemcpyDeviceToHost);

    puts("lec");
    for (int i = 0; i < n_expert; ++i) {
        printf("%d ", lec[i]);
    }
    putchar(10);
}

