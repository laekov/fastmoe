#!/bin/bash
if [ ! -z $OMPI_COMM_WORLD_LOCAL_RANK ]
then
	export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
fi

export PYTHONPATH=$PWD/build/lib.linux-x86_64-3.7
export LD_LIBRARY_PATH=/home/laekov/.local/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
if [ -z $1 ]
then
	python3 moe_test.py 2>logs/$OMPI_COMM_WORLD_RANK.log
elif [ .$1 = '.test_all' ]
then
	for nexp in 1 2 4 
	do
		for inf in 1024 
		do
			for ouf in 4096
			do
				for bs in 4 16 64 256 512 1024 2048 4096
				do
					echo $bs $nexp ${inf}x${ouf} 
					python3 moe_test.py $bs $inf $ouf $nexp
				done
			done
		done
	done
else
	python3 $@ # 2>logs/$OMPI_COMM_WORLD_RANK.log
fi
