#include "../local_exchange.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* args[]) {
    int n_worker = atoi(args[1]);
    int n_expert = atoi(args[2]);
    int batch_size = atoi(args[3]);
    int topk = atoi(args[4]);
    int tot_expert = n_worker * n_expert;

    long* gate_idx = new long[batch_size * topk];
    long* n_gate_idx = new long[batch_size * topk];

    int* lec = new int[tot_expert];
    memset(lec, 0, sizeof(int) * tot_expert);
    for (int i = 0; i < batch_size * topk; ++i) {
        if (rand() % 10) {
            gate_idx[i] = rand() % tot_expert;
            ++lec[gate_idx[i]];
        } else {
            gate_idx[i] = -1;
        }
    }
    for (int i = 1; i < tot_expert; ++i) {
        lec[i] += lec[i - 1];
    }

    puts("gate idx");
    for (int i = 0; i < batch_size * topk; ++i) {
        printf("%d ", gate_idx[i]);
    }
    putchar(10);
    int nlec = lec[tot_expert - 1];

    int* g_lec;
    cudaMalloc(&g_lec, sizeof(int) * tot_expert);
    cudaMemcpy(g_lec, lec, sizeof(int) * tot_expert, cudaMemcpyHostToDevice);
    long* g_gate_idx;
    cudaMalloc(&g_gate_idx, sizeof(long) * batch_size * topk);
    cudaMemcpy(g_gate_idx, gate_idx, sizeof(long) * batch_size * topk,
            cudaMemcpyHostToDevice);
    long* g_pos;
    cudaMalloc(&g_pos, sizeof(long) * nlec);
    // cudaMemcpy(g_gate_idx, gate_idx, sizeof(long) * nlec, cudaMemcpyHostToDevice);

    auto smgr = getCudaStreamManager(0);
    fmoe_cuda_assign_pos_impl(g_lec, g_gate_idx, g_pos, batch_size * topk,
            topk, smgr);

    long* pos = new long[nlec];
    cudaMemcpy(pos, g_pos, sizeof(long) * nlec, cudaMemcpyDeviceToHost);

    puts("pos");
    for (int i = 0; i < nlec; ++i) {
        printf("%d ", pos[i]);
    }
    putchar(10);
}

