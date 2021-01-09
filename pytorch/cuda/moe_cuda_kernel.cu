#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdio>
#include <iostream>
#include <vector>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>                                                                                          
#include <helper_cuda.h> 
#include <c10/cuda/CUDAGuard.h>

#include "timer.hh"

#include "cublas_wrapper.h"
#include "cuda_stream_manager.h"

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)

// #define MOE_BREAKDOWN
// #define MOE_DEBUG

thread_local CudaStreamManager smgr;

template <typename scalar_t>
__global__
void generate_ptr_offset_kernel(size_t n, const scalar_t* base, size_t stride,
		const int* offset, const scalar_t** ptrs) { 
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		ptrs[idx] = base + stride * offset[idx];
	}
}


template <typename scalar_t>
__global__
void batch_scatter_kernel(size_t wid, const int* pos, 
		const scalar_t* inbuf, scalar_t* oubuf) { 
	inbuf += wid * blockIdx.x;
	oubuf += wid * pos[blockIdx.x];
	for (int i = threadIdx.x; i < wid; i += blockDim.x) {
		oubuf[i] = inbuf[i];
	}
}

void moe_cuda_expert_count_impl(
        const int* d_gate,
		int* expert_count,
		int* d_pos,
		const size_t num_expert,
        const size_t batch_size) {
    int *gate = new int[batch_size];
	int *expert_ptr = new int[num_expert];
	memset(expert_count, 0, sizeof(int) * num_expert);

	checkCudaErrors(cudaMemcpy(gate, d_gate, sizeof(int) * batch_size,
				cudaMemcpyDeviceToHost));

	for (int i = 0; i < batch_size; ++i) {
		++expert_count[gate[i]];
	}
	expert_ptr[0] = 0;
	for (int i = 1; i < num_expert; ++i) {
		expert_ptr[i] = expert_ptr[i - 1] + expert_count[i - 1];
	}

	int *pos = new int[batch_size];

	for (int i = 0; i < batch_size; ++i) {
		pos[i] = expert_ptr[gate[i]]++;
	}
	checkCudaErrors(cudaMemcpy(d_pos, pos, sizeof(int) * batch_size,
				cudaMemcpyHostToDevice));
	delete [] gate;
	delete [] expert_ptr;

	ENSURE_SMGR(smgr, num_expert);
}

template <typename scalar_t>
void moe_cuda_local_scatter_impl(
        const scalar_t* input,
		const int* d_pos,
		scalar_t* input_buf,
		const size_t batch_size,
		const size_t in_feat) {
	batch_scatter_kernel<scalar_t>
		<<<batch_size, 256, 0, smgr.streams[0]>>>(in_feat, d_pos, input,
				input_buf); 
	smgr.sync(0);
}

template <typename scalar_t>
__global__
void batch_gather_kernel(size_t wid, const int* pos, 
		const scalar_t* inbuf, scalar_t* oubuf) { 
	inbuf += wid * pos[blockIdx.x];
	oubuf += wid * blockIdx.x;
	for (int i = threadIdx.x; i < wid; i += blockDim.x) {
		oubuf[i] = inbuf[i];
	}
}

template <typename scalar_t>
void moe_cuda_local_gather_impl(
        const scalar_t* output_buf,
		const int* d_pos,
		scalar_t* output,
		const size_t batch_size,
		const size_t out_feat) {
	batch_gather_kernel<scalar_t>
		<<<batch_size, 256, 0, smgr.streams[0]>>>(out_feat, d_pos, output_buf,
				output); 
	smgr.sync(0);
}

template <typename scalar_t>
void moe_cuda_forward_impl(
        const scalar_t* input_buf,
        const scalar_t* weight,
		const int* expert_count,
        scalar_t* output_buf,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert) {
	scalar_t alpha = 1, beta = 0; 

	for (int i = 0, ptr = 0; i < num_expert; ++i) {
		if (expert_count[i] == 0) {
			continue;
		}
		// Use T(B) x T(A) = T(C) to produce row-major C
		checkCudaErrors(cublasXgemm(smgr.handles[i],
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				out_feat, expert_count[i], in_feat,
				&alpha,
				weight + i * in_feat * out_feat, in_feat,
				input_buf + ptr * in_feat, in_feat,
				&beta,
				output_buf + out_feat * ptr, out_feat
				));

		ptr += expert_count[i];
	}
	smgr.sync();
}

template <typename scalar_t>
void moe_cuda_backward_impl(
        const scalar_t* grad_output_buf,
        const scalar_t* input_buf,
		const scalar_t* weight,
		const int* expert_count,
        scalar_t* grad_input_buf,
        scalar_t* grad_weight,
        const size_t batch_size,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert) {
	ENSURE_SMGR(smgr, num_expert);
    scalar_t alpha = 1, beta = 0;

	for (int i = 0, ptr = 0; i < num_expert; ++i) {
		if (expert_count[i] == 0) {
			cudaMemset(grad_weight + i * in_feat * out_feat, 0, 
					sizeof(scalar_t) * in_feat * out_feat);
			continue;
		}
		// Use T(B) x T(A) = T(C) to produce row-major C

		// Backward input: g_i = w @ g_o
		checkCudaErrors(cublasXgemm(smgr.handles[i],
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				in_feat, expert_count[i], out_feat,
				&alpha,
				weight + i * in_feat * out_feat, in_feat,
				grad_output_buf + ptr * out_feat, out_feat,
				&beta,
				grad_input_buf + in_feat * ptr, in_feat
				));

		// Backward weight: g_w = i @ g_o
		checkCudaErrors(cublasXgemm(smgr.handles[i],
				CUBLAS_OP_N,
				CUBLAS_OP_T,
				in_feat, out_feat, expert_count[i],
				&alpha,
				input_buf + in_feat * ptr, in_feat,
				grad_output_buf + ptr * out_feat, out_feat,
				&beta,
				grad_weight + i * in_feat * out_feat, in_feat
				));

		ptr += expert_count[i];
	}
	smgr.sync();
}


std::vector<torch::Tensor> moe_cuda_expert_count(
		torch::Tensor weight,
		torch::Tensor gate) {
	const auto batch_size = gate.size(0);
	const auto num_expert = weight.size(0);

	auto ec_options = torch::TensorOptions().dtype(torch::kInt32);
	auto expert_count = torch::empty(num_expert, ec_options);

	auto pos_options = torch::TensorOptions()
		.device(gate.device())
		.dtype(torch::kInt32);
	auto pos = torch::empty(batch_size, pos_options);
	moe_cuda_expert_count_impl(
			gate.data_ptr<int>(),
			expert_count.data_ptr<int>(),
			pos.data_ptr<int>(),
			num_expert,
			batch_size);

	return {expert_count, pos};
}

std::vector<torch::Tensor> moe_cuda_local_scatter(
    torch::Tensor input,
	torch::Tensor pos) {
	const auto batch_size = input.size(0);
    const auto in_feat = input.size(1);

	auto input_buf = torch::empty_like(input);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_local_scatter_cuda", 
			([&] {
		moe_cuda_local_scatter_impl<scalar_t>(
			input.data_ptr<scalar_t>(),
			pos.data_ptr<int>(),
			input_buf.data_ptr<scalar_t>(),
			batch_size,
			in_feat);
	}));
	return {input_buf,};
}

std::vector<torch::Tensor> moe_cuda_local_gather(
	torch::Tensor output_buf,
	torch::Tensor pos) {
	const auto batch_size = output_buf.size(0);
    const auto out_feat = output_buf.size(1);

	auto output = torch::empty_like(output_buf);

    AT_DISPATCH_FLOATING_TYPES(output_buf.scalar_type(), "moe_local_gather_cuda", 
			([&] {
		moe_cuda_local_gather_impl<scalar_t>(
			output_buf.data_ptr<scalar_t>(),
			pos.data_ptr<int>(),
			output.data_ptr<scalar_t>(),
			batch_size,
			out_feat);
	}));
	return {output,};
}

std::vector<torch::Tensor> moe_cuda_forward(
        torch::Tensor input_buf,
        torch::Tensor weight,
		torch::Tensor expert_count
		) {
	const auto batch_size = input_buf.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);
            
#ifdef MOE_DEBUG
    printf("[forward] expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld\n", 
			num_expert, in_feat, out_feat);
#endif
	/*
    const int device = device_of(input).value().index();
    if (smgr.streams == NULL) {
        smgr.setup(num_expert, device);
    }
	*/
	auto out_options = torch::TensorOptions()
		.device(input_buf.device())
		.dtype(input_buf.dtype());
    auto output = torch::empty({batch_size, out_feat}, out_options);
    
    AT_DISPATCH_FLOATING_TYPES(input_buf.scalar_type(), "moe_forward_cuda", 
			([&] {
		moe_cuda_forward_impl<scalar_t>(
			input_buf.data_ptr<scalar_t>(),
			weight.data_ptr<scalar_t>(),
			expert_count.data_ptr<int>(),
			output.data_ptr<scalar_t>(),
			in_feat,
			out_feat,
			num_expert
		);
    }));
    
    return {output, };           
}

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output_buf, // [batch_size x out_feat]
    torch::Tensor input_buf, // [batch_size x out_feat]
    torch::Tensor weight, // [num_expert x out_feat x in_feat]
	torch::Tensor expert_count
) {
    const auto batch_size = input_buf.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);

#ifdef MOE_DEBUG
    printf("[backward] b=%ld, expert=%ld, in_feat (d_model)=%ld, "
			"out_feat (d_ffn)=%ld\n",
			batch_size, num_expert, in_feat, out_feat);
#endif

    auto grad_input_buf = grad_output_buf.new_empty({batch_size, in_feat}); 
    auto grad_weight = grad_output_buf.new_empty({num_expert, out_feat, in_feat});

    AT_DISPATCH_FLOATING_TYPES(input_buf.scalar_type(), "moe_cuda_backward", ([&] {
        moe_cuda_backward_impl<scalar_t>(
            grad_output_buf.data_ptr<scalar_t>(),
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
			expert_count.data_ptr<int>(),
            grad_input_buf.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            batch_size,
            in_feat,
            out_feat,
            num_expert
        );
    }));

    return {grad_input_buf, grad_weight};
}


/*
int main() {
    typedef float data_t;
    size_t batch_size = 4096;
    size_t top_k = 2;
    size_t num_expert = 128;
    size_t in_feat = 1024;
    size_t out_feat = 4096;
	data_t *input, *weight;
	data_t *output;
	size_t *gate;

	checkCudaErrors(cudaMalloc(&input, batch_size * in_feat * sizeof(data_t)));
	checkCudaErrors(cudaMalloc(&weight, num_expert * in_feat * out_feat * sizeof(data_t)));	
	checkCudaErrors(cudaMalloc(&output, batch_size * top_k * out_feat * sizeof(data_t)));
    checkCudaErrors(cudaMalloc(&gate, batch_size * top_k * sizeof(size_t)));
    
    size_t nt = 16;
    double tsum = 0, tmax = 0;

    size_t *gate_host = new size_t[batch_size * top_k];
    for (size_t i=0; i<batch_size * top_k; ++i) {
        gate_host[i] = rand() % num_expert;
    } 
    checkCudaErrors(cudaMemcpy(gate, gate_host, batch_size * top_k * sizeof(size_t), cudaMemcpyHostToDevice));

    moe_first_linear_cuda_forward<data_t>(input, gate, weight, output, batch_size, top_k, in_feat, out_feat);
    
    for (size_t i=0; i<nt; ++i) {
        timestamp(start);
		moe_first_linear_cuda_forward<data_t>(input, gate, weight, output, batch_size, top_k, in_feat, out_feat);
		timestamp(end);
		auto t = getDuration(start, end);
		tsum += t;
		if (t > tmax) tmax = t;
    }
    printf("Mean %.3lf us, max %.3lf us\n", tsum / nt * 1e6, tmax * 1e6);
	double tflops = (double)batch_size * top_k * in_feat * out_feat * nt * 2e-12 / tsum;
	printf("%.3lf TFLOPs\n", tflops);
}
*/
