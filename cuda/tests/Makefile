default : test_balancing

test_% : %.cu
	nvcc $< ../stream_manager.cpp -lcublas -o $@
