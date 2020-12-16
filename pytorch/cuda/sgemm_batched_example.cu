#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "timer.hh"

static const char *geterr(cublasStatus_t error)
{
	switch (error)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

#define cublas_safe_call(__fn__) { \
	cublasStatus_t res = __fn__; \
	if (res != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "Cublas " << geterr(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
	} \
}

#define cuda_safe_call(__fn__) { \
	auto res = __fn__; \
	if (res) { \
		std::cerr << "CUDA" << cudaGetErrorString(res) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
	} \
}

using namespace std;

typedef float data_t;

int d_batch = 4096;
int d_matx = 1;
int d_experts = 128;
int d_in = 1024;
int d_hidden = 4096;
int d_out = 1024;

data_t *featin, *feath, *weight1, *weight2, *featout;
int *offset;

cublasHandle_t hdl;
cudaStream_t st;

void prepare() {
	cudaStreamCreate(&st);
	cublasCreate(&hdl);
	cublasSetStream(hdl, st);
}


void compute() {
	vector<data_t*> aptrs, bptrs, cptrs;
	float **ptrs;
	cudaMalloc(&ptrs, d_batch * sizeof(float*) * 3);
	for (int i = 0; i < d_batch; ++i) {
		aptrs.push_back(featin + 1 * d_in * i);
		bptrs.push_back(weight1 + d_hidden * d_in * offset[i]);
		cptrs.push_back(feath + d_hidden * i);
	}
	cudaMemcpy(ptrs, aptrs.data(), d_batch * sizeof(float*), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(ptrs + d_batch, bptrs.data(), d_batch * sizeof(float*), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(ptrs + d_batch * 2, cptrs.data(), d_batch * sizeof(float*), 
			cudaMemcpyHostToDevice);
	data_t alpha = 1, beta = 0;
	cublas_safe_call(cublasSgemmBatched(hdl, 
			CUBLAS_OP_T,
			CUBLAS_OP_T,
			d_matx, d_hidden, d_in,
			&alpha,
			ptrs, d_in,
			ptrs + d_batch, d_hidden,
			&beta,
			ptrs + d_batch * 2, d_matx,
			d_batch));
	cudaStreamSynchronize(st);
	// cudaDeviceSynchronize();
}

int main() {
	cuda_safe_call(cudaMalloc(&weight1, d_in * d_hidden * d_experts * sizeof(data_t)));
	cudaMalloc(&weight2, d_out * d_hidden * d_experts * sizeof(data_t));
	cudaMalloc(&featin, d_batch * d_matx * d_in * sizeof(data_t));
	cudaMalloc(&feath, d_batch * d_matx * d_hidden * sizeof(data_t));
	cudaMalloc(&featout, d_batch * d_matx * d_out * sizeof(data_t));

	prepare();

	double tsum = 0, tmax = 0;
	int nt = 16;
	offset = new int[d_batch];
	for (int i = 0; i < d_batch; ++i) {
		offset[i] = rand() % d_experts;
	}
	compute();
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < d_batch; ++j) {
			offset[j] = rand() % d_experts;
		}
		timestamp(start);
		compute();
		timestamp(end);
		auto t = getDuration(start, end);
		tsum += t;
		if (t > tmax) tmax = t;
	}
	printf("Mean %.3lf us, max %.3lf us\n", tsum / nt * 1e6, tmax * 1e6);
	double tflops = (double)d_batch * d_matx * d_in * (double)d_hidden * nt * 2e-12 / tsum;
	printf("%.3lf TFLOPs\n", tflops);
}
