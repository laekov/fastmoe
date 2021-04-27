#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>

std::vector<torch::Tensor> _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return {global_expert_count};
}

std::vector<torch::Tensor> _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr
        );
    }));
    return {global_input_buf,};
}

std::vector<torch::Tensor> _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(), 
            "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr
        );
    }));
    return {local_output_buf,};
}

#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
        ncclComm_t comm;
        ncclCommInitRank(&comm, getSize(), ncclID, rank);
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
