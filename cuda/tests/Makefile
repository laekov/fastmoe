default : test_prune_gate test_limit test_assign test_counting

test_% : %.cu
	nvcc $< ../stream_manager.cpp -lcublas -o $@
