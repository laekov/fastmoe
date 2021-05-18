#include "local_exchange.cuh"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

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

void _assign_pos(
    torch::Tensor cum_count,
    torch::Tensor gate,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(cum_count.device().index());
    auto gate_shp = gate.sizes();
    size_t batch_size = gate_shp[0], topk = 1;
    if (gate_shp.size() == 2) {
        topk = gate_shp[1];
    }
    fmoe_cuda_assign_pos_impl(
            cum_count.data_ptr<int>(),
            gate.data_ptr<long>(),
            pos.data_ptr<long>(),
            batch_size, topk, smgr);
}
