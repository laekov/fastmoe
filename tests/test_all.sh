#!/bin/bash

runtest() {
	echo Testing $@
	$@
	if [ $? = 0 ]
	then
		echo '----------------- Passed'
	else
		echo '----------------- Failed'
		exit
	fi
}

if [ ! -z $1 ]
then
	runtest $@
	exit
fi

TEST_SCRIPT=$(dirname $(realpath $0))/test.sh
runtest $TEST_SCRIPT tests/test_numerical.py
runtest mpirun -n 2 $TEST_SCRIPT tests/test_numerical.py
runtest $TEST_SCRIPT tests/test_dp.py
runtest $TEST_SCRIPT tests/test_performance.py
runtest mpirun -n 2 $TEST_SCRIPT tests/test_performance.py
