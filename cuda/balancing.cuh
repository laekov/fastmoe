#include "stream_manager.h"
#include "utils/fmoe_utils.h"
#include <cuda.h>

__global__
void limit_by_capacity_kernel(const long* ec, int* cap, long* eca,
		const long n_expert, const long n_worker) {
	int eid = blockIdx.y;
	int wid = blockIdx.x * blockDim.x + threadIdx.x;
	if (wid < n_worker) {
		int proposal = ec[wid * n_expert + eid];
		int cap_left = atomicSub(cap + eid, proposal);
		if (cap_left >= proposal) {
			eca[wid * n_expert + eid] = proposal;
		} else if (cap_left >= 0) {
			eca[wid * n_expert + eid] = cap_left;
		} else {
			eca[wid * n_expert + eid] = 0;
		}
	}
}

void fmoe_cuda_limit_by_capacity_impl(const long* ec, int* cap,
		long* eca, const long n_expert, const long n_worker,
		CudaStreamManager* smgr) {
	dim3 grid_dim(CEIL(n_worker, 1024), n_expert);
	dim3 block_dim(1024);
	limit_by_capacity_kernel<<<grid_dim, block_dim, 0, smgr->stream(0)>>>(
			ec, cap, eca, n_expert, n_worker);
	smgr->sync(1);
}
