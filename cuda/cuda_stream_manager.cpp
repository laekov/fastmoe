#include <unordered_map>
#include <mutex>
#include <cassert>
#include <thread>
#include <iostream>

#ifdef MOE_USE_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif  // MOE_USE_NCCL

#include "cuda_stream_manager.h"
#include <helper_cuda.h> 

#define SMGR_N_STREAMS 16

#ifdef MOE_USE_NCCL
class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
	ncclComm_t getcomm(at::Device dev) {
		auto key = std::to_string(dev.index());
		auto v = getNCCLComm(key, {dev});
		if (v.size() == 0) {
			std::cerr << "PyTorch has nothing\n";
			return 0;
		}
		return v[0]->getNcclComm();
	}
};

void CudaStreamManager::ensure(void* torchp, at::Device dev) {
	if (this->ncclgood) {
		return;
	}
	HackNCCLGroup* h = (HackNCCLGroup*)torchp;
	this->ncclcomm = h->getcomm(dev);
	if (this->ncclcomm != 0) {
		this->ncclgood = 1;
	} else {
		std::cerr << "Nccl initialization failed\n";
	}
}
#endif  // MOE_USE_NCCL

cudaStream_t CudaStreamManager::stream(size_t idx) {
	return this->streams[idx % SMGR_N_STREAMS];
}

cublasHandle_t CudaStreamManager::handle(size_t idx) {
	return this->handles[idx % SMGR_N_STREAMS];
}


void CudaStreamManager::sync(int idx) {
	for (int i = 0; i < idx && i < SMGR_N_STREAMS; ++i) {
		cudaStreamSynchronize(streams[i]);
	}
}

void CudaStreamManager::setup(const int device) {
#ifdef MOE_USE_NCCL
	this->ncclgood = 0;
#endif
	this->device = device;
	checkCudaErrors(cudaSetDevice(device));
	streams = new cudaStream_t[SMGR_N_STREAMS];
	handles = new cublasHandle_t[SMGR_N_STREAMS];
	for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
		checkCudaErrors(cudaStreamCreate(streams + i));
		checkCudaErrors(cublasCreate(handles + i));
		cublasSetStream(handles[i], streams[i]);
	}
}

void CudaStreamManager::destroy() {
	for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
		checkCudaErrors(cudaStreamDestroy(streams[i]));
		checkCudaErrors(cublasDestroy(handles[i]));
	}
	delete[] streams;
	delete[] handles;
}

std::unordered_map<int, CudaStreamManager*> smgrs;
std::mutex smgr_mtx;

CudaStreamManager* getCudaStreamManager(const int device) {
	auto it = smgrs.find(device);
	if (it == smgrs.end()) {
		smgr_mtx.lock();
		it = smgrs.find(device);
		if (it == smgrs.end()) {
			auto smgr = new CudaStreamManager(device);
			smgrs.insert(std::pair<int, CudaStreamManager*>(device, smgr));
			smgr_mtx.unlock();
			return smgr;
		} else {
			smgr_mtx.unlock();
		}
	}
	return it->second;
}

