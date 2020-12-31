#include "comm_manager.h"


CommManager* comm_mgr = 0;


CommManager* getCommManager() {
	if (!comm_mgr) {
		comm_mgr = new CommManager();
	}
	return comm_mgr;
}

