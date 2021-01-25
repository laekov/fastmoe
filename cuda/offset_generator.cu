#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>

using namespace std;

typedef float data_t;

cudaStream_t st = 0;

__global__
void generate_ptr_sequential_kernel(int n, data_t* base, size_t stride, data_t** ptrs) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		ptrs[idx] = base + stride * idx;
	}
}

__global__
void generate_ptr_offset_kernel(int n, data_t* base, size_t stride, int* offset, data_t** ptrs) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		ptrs[idx] = base + stride * offset[idx];
	}
}

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)

data_t** generate_ptr(int n, data_t* base, size_t stride, int* d_offset = 0) {
	dim3 griddim(CEIL(n, 256));
	dim3 blockdim(256);
	data_t** ptrs;
	cudaMalloc(&ptrs , n * sizeof(data_t*));
	if (d_offset) {
		generate_ptr_offset_kernel<<<griddim, blockdim, 0, st>>>(n, base, stride, d_offset, ptrs);
	} else {
		generate_ptr_sequential_kernel<<<griddim, blockdim, 0, st>>>(n, base, stride, ptrs);
	}
	cudaError_t err = cudaPeekAtLastError();
	if (err) {
		std::cerr << "CUDA" << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;
	}
	cudaStreamSynchronize(st);
	return ptrs;
}

int main() {
	cudaStreamCreate(&st);
	int n = 128;
	int offset[128], *d_offset;
	float* base = (float*)0x10, **d_res, **res;
	for (int i = 0; i < n; ++i) {
		offset[i] = rand() % n;
	}
	cudaMalloc(&d_offset, n * sizeof(int));
	cudaMemcpy(d_offset, offset, n * sizeof(int), cudaMemcpyHostToDevice);

	d_res = generate_ptr(n, base, 0x100);
	res = new float*[n];
	cudaMemcpy(res, d_res, n * sizeof(float*), cudaMemcpyDeviceToHost);
	puts("Sequential addr check");
	for (int i = 0; i < 10; ++i) {
		printf("%08x  ", (unsigned long)res[i]);
	}
	putchar(10);

	d_res = generate_ptr(n, base, 0x400, d_offset);
	res = new float*[n];
	cudaMemcpy(res, d_res, n * sizeof(float*), cudaMemcpyDeviceToHost);
	puts("Sequential addr check");
	for (int i = 0; i < 10; ++i) {
		printf("%08x  /%08x\n", (unsigned long)res[i], offset[i]);
	}
	putchar(10);

}
