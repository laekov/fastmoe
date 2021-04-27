#include "balancing.cuh"
#include <torch/extension.h>

/* 
 * note that due to limit of cuda atomic operator, capacity should be int32
 */
std::vector<torch::Tensor> _limit_by_capacity(
		torch::Tensor expert_count, torch::Tensor capacity,
		long n_expert, long n_experts) {
    auto expert_count_ack = torch::empty_like(expert_count);
    auto smgr = getCudaStreamManager(expert_count.device().index());
	fmoe_cuda_limit_by_capacity_impl(
			expert_count.data_ptr<long>(),
			capacity.data_ptr<int>(),
			expert_count_ack.data_ptr<long>(),
			n_expert, n_workers, smgr);
}
