#include "../balancing.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char* args[]) {
    int n_worker = atoi(args[1]);
    int n_expert = atoi(args[2]);
    int cap_v = atoi(args[3]);
    int tot_expert = n_worker * n_expert;

    long* lec = new long[tot_expert];
    for (int i = 0; i < tot_expert; ++i) {
        lec[i] = i;
    }
    long* g_lec;
    cudaMalloc(&g_lec, sizeof(long) * tot_expert);
    cudaMemcpy(g_lec, lec, sizeof(long) * tot_expert, cudaMemcpyHostToDevice);

    int* cap = new int[n_expert];
    for (int i = 0; i < n_expert; ++i) {
        cap[i] = cap_v;
    }
    int* g_cap;
    cudaMalloc(&g_cap, sizeof(int) * n_expert);
    cudaMemcpy(g_cap, cap, sizeof(int) * n_expert, cudaMemcpyHostToDevice);

    long* eca = new long[tot_expert];
    long* g_eca;
    cudaMalloc(&g_eca, sizeof(long) * tot_expert);

    auto smgr = getCudaStreamManager(0);
    fmoe_cuda_limit_by_capacity_impl(g_lec, g_cap, g_eca, n_expert, n_worker, smgr);

    cudaMemcpy(cap, g_cap, sizeof(int) * n_expert, cudaMemcpyDeviceToHost);
    cudaMemcpy(eca, g_eca, sizeof(long) * tot_expert, cudaMemcpyDeviceToHost);

    printf("%d\n", cap[0]);
    for (int i = 0; i < tot_expert; ++i) {
        printf("%ld %ld\n", lec[i], eca[i]);
    }
}
