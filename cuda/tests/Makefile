default : test_prune_gate test_limit

test_% : %.cu
	nvcc $< ../stream_manager.cpp -lcublas -o $@
