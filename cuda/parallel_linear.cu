#include "parallel_linear.cuh"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

torch::Tensor _linear_forward(
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        ) {
    auto smgr = getCudaStreamManager(input_buf.device().index());
    const auto batch_size = input_buf.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);

#ifdef MOE_DEBUG
    printf("[forward] expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld\n",
            num_expert, in_feat, out_feat);
#endif

    torch::Tensor output;

    if (bias.has_value()) {
        output = bias.value().repeat_interleave(expert_count.to(bias.value().device()), 0);
    } else{
        auto out_options = torch::TensorOptions()
            .device(input_buf.device())
            .dtype(input_buf.dtype());
        output = torch::empty({batch_size, out_feat}, out_options);
    }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), "moe_forward_cuda",
            ([&] {
        fmoe_cuda_linear_forward_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            expert_count.data_ptr<long>(),
            output.data_ptr<scalar_t>(),
            bias.has_value(),
            in_feat,
            out_feat,
            num_expert,
            smgr
        );
    }));

    return output;
}


std::vector<torch::Tensor> _linear_backward(
    torch::Tensor grad_output_buf,
    torch::Tensor input_buf,
    torch::Tensor expert_count,
    torch::Tensor weight,
    at::optional<torch::Tensor> bias
) {
    auto smgr = getCudaStreamManager(input_buf.device().index());
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
    auto grad_bias = grad_output_buf.new_empty({num_expert, out_feat});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), "moe_cuda_backward", ([&] {
        fmoe_cuda_linear_backward_impl<scalar_t>(
            grad_output_buf.data_ptr<scalar_t>(),
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            expert_count.data_ptr<long>(),
            grad_input_buf.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            bias.has_value(),
            batch_size,
            in_feat,
            out_feat,
            num_expert,
            smgr
        );
    }));

    return {grad_input_buf, grad_weight, grad_bias};
}

