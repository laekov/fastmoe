#ifdef FMOE_USE_NCCL

#include <cstdlib>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "smart_schedule.h"

long pipeline_gran = -1;

torch::Tensor _smart_sch_forward(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long global_batch_size,
        long n_workers,
        py::function forward_fn) {

    if (pipeline_gran == -1) {
        char* p = getenv("FMOE_FASTER_GROUP_SIZE");
        if (p) {
            pipeline_gran = atoi(p);
        } else {
            pipeline_gran = 4;
        }
    }

    auto smgr = getCudaStreamManager(input_buf.device().index());
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    const auto num_expert = local_expert_count.size(0) / n_workers;
    const auto d_model = input_buf.size(1);

    auto global_input_buf = input_buf.new_zeros({global_batch_size, d_model});
    auto global_output_buf = input_buf.new_zeros({global_batch_size, d_model});
    
    auto output_buf = input_buf.new_zeros({input_buf.size(0), d_model});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_fused_forward", ([&] {
        fmoe_cuda_fused_forward_impl(
            forward_fn,
            input_buf.device(),

            input_buf.data_ptr<scalar_t>(),
            global_input_buf.data_ptr<scalar_t>(),
            global_output_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            stored_models.data_ptr<bool>(),
            d_model, num_expert, rank, n_workers,
            pipeline_gran, smgr);
    }));
    return output_buf;
}

/*
std::vector<torch::Tensor> _fused_backward(
        torch::Tensor input_buf,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,
        torch::Tensor middle_buf,
        torch::Tensor output_buf,
        torch::Tensor grad_out,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor inp,
        torch::Tensor stored_models,
        
        long global_batch_size,
        long buf_batch_size,
        long n_workers, bool has_bias) {
    const auto num_expert = local_expert_count.size(0) / n_workers;
    
    auto smgr = getCudaStreamManager(input_buf.device().index());
    int rank;
    ncclCommUserRank(smgr->ncclcomm, &rank);
    
    const auto d_hidden = params[rank][0][0].size(1);
    const auto d_model = params[rank][0][0].size(2);


    auto global_grad_out = input_buf.new_zeros({global_batch_size, d_model});
    auto grad_middle = input_buf.new_zeros({global_batch_size, d_hidden});
    auto global_grad_in = input_buf.new_zeros({global_batch_size, d_model});
    
    auto grad_in = input_buf.new_zeros({buf_batch_size, d_model});
    
    for (auto node : params)
        for (auto expert : node)
            for (int i = 0; i < expert.size(); i++) {
                // create the respective gradient of each tensor
                CHECK_INPUT(expert[i]);
                if (expert[i].grad().defined()) {
                    CHECK_INPUT(expert[i].grad());
                    continue;
                }

                expert[i].mutable_grad() = input_buf.new_zeros(expert[i].sizes());
            }

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_fused_backward", ([&] {
        fmoe_cuda_fused_backward_impl(
            input_buf.data_ptr<scalar_t>(),
            inp.data_ptr<scalar_t>(),
            params,

            middle_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),

            global_grad_out.data_ptr<scalar_t>(),
            global_grad_in.data_ptr<scalar_t>(),

            grad_middle.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            stored_models.data_ptr<bool>(),
            d_model, d_hidden, num_expert, rank, n_workers, has_bias,
            pipeline_gran, smgr);
    }));
    return {grad_in,};
}
*/
#endif

