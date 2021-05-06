#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <vector>

#include "moe_cuda_kernel.h"

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> moe_expert_count(
        torch::Tensor gate, 
        size_t num_expert) {
    CHECK_INPUT(gate);
    return moe_cuda_expert_count(gate, num_expert);
}

std::vector<torch::Tensor> moe_local_scatter(
        torch::Tensor input,
        torch::Tensor pos) {
    CHECK_INPUT(input);
    return moe_cuda_local_scatter(input, pos);
}

std::vector<torch::Tensor> moe_local_gather(
        torch::Tensor output_buf,
        torch::Tensor pos) {
    CHECK_INPUT(output_buf);
    return moe_cuda_local_gather(output_buf, pos);
}

std::vector<torch::Tensor> moe_forward(
        torch::Tensor input_buf,     		// [batch_size x in_feat]
        torch::Tensor expert_count,  		// [num_expert]
        torch::Tensor weight,        		// [num_expert x out_feat x in_feat]
        at::optional<torch::Tensor> bias_o  // [num_expert x out_feat] or None
        ) {

    CHECK_INPUT(input_buf);
    CHECK_INPUT(weight);
    
    // check if bias is valid in case it exists
    if (bias_o.has_value()) {
        auto bias = bias_o.value();
        CHECK_INPUT(bias);
    }
    
    return moe_cuda_forward(input_buf, expert_count, weight, bias_o);
}

std::vector<torch::Tensor> moe_backward(
        torch::Tensor grad_output_buf, 		// [batch_size x out_feat]
        torch::Tensor input_buf,       		// [batch_size x in_feat]
        torch::Tensor expert_count,         // [num_expert]
        torch::Tensor weight,           	// [num_expert x out_feat x in_feat]
        at::optional<torch::Tensor> bias_o  // [num_expert x out_feat] or None
        ) {
    
    CHECK_INPUT(grad_output_buf);
    CHECK_INPUT(input_buf);
    CHECK_INPUT(weight);

    // check if bias is valid in case it exists
    if (bias_o.has_value()) {
        auto bias = bias_o.value();
        CHECK_INPUT(bias);
    }
    
    return moe_cuda_backward(grad_output_buf, input_buf, expert_count, weight, bias_o);
}

#ifdef MOE_USE_NCCL

std::vector<torch::Tensor> moe_expert_exchange(
        torch::Tensor local_expert_count,
        size_t num_expert, size_t n_workers) {
    return moe_cuda_expert_exchange(local_expert_count, num_expert, n_workers);
}

std::vector<torch::Tensor> moe_global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        size_t batch_size, size_t n_workers) {
    CHECK_INPUT(input_buf);
    return moe_cuda_global_scatter(input_buf,
                local_expert_count, global_expert_count,
                batch_size, n_workers);
}

std::vector<torch::Tensor> moe_global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        size_t batch_size, size_t n_workers) {
    CHECK_INPUT(output_buf);
    return moe_cuda_global_gather(output_buf,
                local_expert_count, global_expert_count,
                batch_size, n_workers);
}


std::vector<torch::Tensor> moe_global_fused_forward(
        torch::Tensor input_buf,
        torch::Tensor weight,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long global_batch_size, long local_batch_size, long n_workers) {
    CHECK_INPUT(input_buf);
    CHECK_INPUT(weight);
    return moe_cuda_global_fused_forward(
            input_buf, weight, local_expert_count, global_expert_count,
            global_batch_size, local_batch_size, n_workers);
}

#include <c10d/ProcessGroupNCCL.hpp>
#include "cuda_stream_manager.h"

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        auto key = std::to_string(dev.index());
#ifdef ENABLE_NCCL_P2P_SUPPORT
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
#else
        auto v = getNCCLComm(key, {dev});
        if (v.size() == 0) {
            std::cerr << "PyTorch has nothing\n";
            return 0;
        }
        int count;
        ncclCommCount(v[0]->getNcclComm(), &count);
        std::cerr << "PyTorch has " << v.size() << " comms, comm 0 size " << count << "\n";
        return v[0]->getNcclComm();
#endif
    }
};

void moe_ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
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

#endif  // MOE_USE_NCCL

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert_count", &moe_expert_count, "MoE expert count (CUDA)");
  m.def("local_scatter", &moe_local_scatter, "MoE local scatter (CUDA)");
  m.def("local_gather", &moe_local_gather, "MoE local gather (CUDA)");
#ifdef MOE_USE_NCCL
  m.def("expert_exchange", &moe_expert_exchange, "MoE expert exchange (CUDA)");
  m.def("global_scatter", &moe_global_scatter, "MoE global scatter (CUDA)");
  m.def("global_gather", &moe_global_gather, "MoE global gather (CUDA)");
  m.def("global_fused_forward", &moe_global_fused_forward, 
          "MoE global gather (CUDA)");
  m.def("ensure_nccl", &moe_ensure_nccl, "MoE ensure torch nccl comm");
#endif
  m.def("forward", &moe_forward, "MoE forward (CUDA)");
  m.def("backward", &moe_backward, "MoE backward (CUDA)");
}