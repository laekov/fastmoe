#include "moe_cuda_kernel.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_stream_manager.h"

#ifdef MOE_USE_NCCL
#include <nccl.h>

void moe_cuda_expert_exchange_impl(
		const long* local_expert_count, 
		long* global_expert_count, 
		int num_expert, int world_size,
		CudaStreamManager* smgr) {
	NCCL_SAFE_CALL(ncclGroupStart());
	for (int i = 0; i < world_size; ++i) {
		NCCL_SAFE_CALL(ncclSend(
				local_expert_count + num_expert * i,
				num_expert,
				ncclInt64,
				i,
				smgr->ncclcomm,
				smgr->stream(0)));
		NCCL_SAFE_CALL(ncclRecv(
				global_expert_count + num_expert * i,
				num_expert,
				ncclInt64,
				i,
				smgr->ncclcomm,
				smgr->stream(0)));
	}
	NCCL_SAFE_CALL(ncclGroupEnd());
	smgr->sync(1);
}

std::vector<torch::Tensor> moe_cuda_expert_exchange(
		torch::Tensor local_expert_count,
		long num_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
	auto smgr = getCudaStreamManager(local_expert_count.device().index());

	moe_cuda_expert_exchange_impl(
			local_expert_count.data_ptr<long>(),
			global_expert_count.data_ptr<long>(),
			num_expert, n_workers,
			smgr);
	return {global_expert_count};
}

template<typename scalar_t>
void moe_cuda_global_scatter_impl(
	const scalar_t* local_input_buf,
	const long* local_expert_count,
	const long* global_expert_count,
	scalar_t* input_buf,
	size_t in_feat, size_t num_expert, size_t world_size,
	CudaStreamManager* smgr) {
	// assert world_size > 1
	int recv_ptr = 0;
	/* TODO: may save for backward */
	long*expert_ptr = new long[num_expert * world_size];
	expert_ptr[0] = 0;
	for (int i = 1; i < num_expert * world_size; ++i) {
		expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
	}

	for (int i = 0; i < num_expert; ++i) {
		NCCL_SAFE_CALL(ncclGroupStart());
		for (int j = 0; j < world_size; ++j) {
			int idx = i + j * num_expert;
			if (local_expert_count[idx]) {
				NCCL_SAFE_CALL(ncclSend(
						local_input_buf + expert_ptr[idx] * in_feat, 
						local_expert_count[idx] * in_feat * sizeof(scalar_t),
						ncclChar, 
						j,
						smgr->ncclcomm,
						smgr->stream(0)));
			}
			if (global_expert_count[idx]) {
				NCCL_SAFE_CALL(ncclRecv(
						input_buf + recv_ptr * in_feat,
						global_expert_count[idx] * in_feat * sizeof(scalar_t),
						ncclChar,
						j,
						smgr->ncclcomm,
						smgr->stream(0)));
				recv_ptr += global_expert_count[idx];
			}
		}
		NCCL_SAFE_CALL(ncclGroupEnd());
	}
	delete [] expert_ptr;
	smgr->sync(1);
}

std::vector<torch::Tensor> moe_cuda_global_scatter(
		torch::Tensor input_buf,
		torch::Tensor local_expert_count,
		torch::Tensor global_expert_count,
		long batch_size, long n_workers) {
	auto num_expert = local_expert_count.size(0) / n_workers;
	auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
	auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
			"moe_cuda_global_scatter", ([&] {
		moe_cuda_global_scatter_impl<scalar_t>(
			input_buf.data_ptr<scalar_t>(),
			local_expert_count.data_ptr<long>(),
			global_expert_count.data_ptr<long>(),
			global_input_buf.data_ptr<scalar_t>(),
			in_feat, num_expert, n_workers,
			smgr
		);
	}));
	return {global_input_buf,};
}

template<typename scalar_t>
void moe_cuda_global_gather_impl(
	const scalar_t* output_buf,
	const long* local_expert_count,
	const long* global_expert_count,
	scalar_t* local_output_buf,
	size_t out_feat, size_t num_expert, size_t world_size,
	CudaStreamManager* smgr) {
	long send_ptr = 0;
	/* TODO: may save for backward */
	long *expert_ptr = new long[num_expert * world_size];
	expert_ptr[0] = 0;
	for (int i = 1; i < num_expert * world_size; ++i) {
		expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
	}

	for (int i = 0; i < num_expert; ++i) {
		NCCL_SAFE_CALL(ncclGroupStart());
		for (int j = 0; j < world_size; ++j) {
			int idx = i + j * num_expert;
			if (global_expert_count[idx]) {
				NCCL_SAFE_CALL(ncclSend(
						output_buf + send_ptr * out_feat,
						global_expert_count[idx] * out_feat * sizeof(scalar_t),
						ncclChar,
						j,
						smgr->ncclcomm,
						smgr->stream(0)));
				send_ptr += global_expert_count[idx];
			}
			if (local_expert_count[idx]) {
				NCCL_SAFE_CALL(ncclRecv(
						local_output_buf + expert_ptr[idx] * out_feat, 
						local_expert_count[idx] * out_feat * sizeof(scalar_t),
						ncclChar, 
						j,
						smgr->ncclcomm,
						smgr->stream(0)));
			}
		}
		NCCL_SAFE_CALL(ncclGroupEnd());
	}
	delete [] expert_ptr;
	smgr->sync(1);
}

std::vector<torch::Tensor> moe_cuda_global_gather(
		torch::Tensor output_buf,
		torch::Tensor local_expert_count,
		torch::Tensor global_expert_count,
		long batch_size, long n_workers) {
	auto num_expert = local_expert_count.size(0) / n_workers;
	auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
	auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(), 
			"moe_cuda_global_gather", ([&] {
		moe_cuda_global_gather_impl<scalar_t>(
			output_buf.data_ptr<scalar_t>(),
			local_expert_count.data_ptr<long>(),
			global_expert_count.data_ptr<long>(),
			local_output_buf.data_ptr<scalar_t>(),
			out_feat, num_expert, n_workers,
			smgr
		);
	}));
	return {local_output_buf,};
}

#endif
