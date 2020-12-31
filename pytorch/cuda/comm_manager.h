#ifndef COMM_MANAGER_H
#define COMM_MANAGER_H

#include <mpi.h>
#include "nccl.h"

struct CommManager {
	int rank, size;
	ncclComm_t ncclcomm;

	CommManager() {
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);

		ncclUniqueId uid;
		if (rank == 0) {
			ncclGetUniqueId(&uid);
		}
		MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
		ncclCommInitRank(&ncclcomm, size, uid, rank);
	}
};

CommManager* getCommManager();

#endif  // COMM_MANAGER
