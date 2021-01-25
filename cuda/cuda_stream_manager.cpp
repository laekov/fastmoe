#include <unordered_map>
#include <mutex>
#include <cassert>
#include <thread>

#include "cuda_stream_manager.h"
#include <helper_cuda.h> 

#define SMGR_N_STREAMS 16

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
	checkCudaErrors(cudaSetDevice(device));
	streams = new cudaStream_t[SMGR_N_STREAMS];
	handles = new cublasHandle_t[SMGR_N_STREAMS];
	for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
		checkCudaErrors(cudaStreamCreate(streams + i));
		checkCudaErrors(cublasCreate(handles + i));
		cublasSetStream(handles[i], streams[i]);
	}
#ifdef MOE_USE_NCCL
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	ncclUniqueId uid;
	if (rank == 0) {
		ncclGetUniqueId(&uid);
	}
	MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
	NCCL_SAFE_CALL(ncclCommInitRank(&ncclcomm, size, uid, rank));
#endif
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

