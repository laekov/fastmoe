#include "local_exchange.cuh"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

std::vector<torch::Tensor> _expert_count(
        torch::Tensor gate, 
        size_t num_expert) {
    const auto batch_size = gate.size(0);

    auto ec_options = torch::TensorOptions().dtype(torch::kInt32);
    auto expert_count = torch::empty(num_expert, ec_options);

    auto pos_options = torch::TensorOptions()
        .device(gate.device())
        .dtype(torch::kInt32);
    auto pos = torch::empty(batch_size, pos_options);
    fmoe_cuda_expert_count_impl(
            gate.data_ptr<int>(),
            expert_count.data_ptr<int>(),
            pos.data_ptr<int>(),
            num_expert,
            batch_size);

    return {expert_count, pos};
}

std::vector<torch::Tensor> _local_scatter(
    torch::Tensor input,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(input.device().index());
    const auto batch_size = pos.size(0);
    const auto in_feat = input.size(1);

    auto opt = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto input_buf = torch::empty({batch_size, in_feat}, opt);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fmoe_local_scatter", 
            ([&] {
        fmoe_cuda_local_scatter_impl<scalar_t>(
            input.data_ptr<scalar_t>(),
            pos.data_ptr<long>(),
            input_buf.data_ptr<scalar_t>(),
            batch_size,
            in_feat,
            smgr);
    }));
    return {input_buf,};
}

std::vector<torch::Tensor> _local_gather(
    torch::Tensor output_buf,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(output_buf.device().index());
    const auto batch_size = pos.size(0);
    const auto out_feat = output_buf.size(1);

    auto opt = torch::TensorOptions()
        .dtype(output_buf.dtype())
        .device(output_buf.device());
    auto output = torch::empty({batch_size, out_feat}, opt);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(), "fmoe_local_gather", 
            ([&] {
        fmoe_cuda_local_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            pos.data_ptr<long>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            out_feat,
            smgr);
    }));
    return {output,};
}
