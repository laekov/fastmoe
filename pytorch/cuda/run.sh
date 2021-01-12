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
else
	python3 $@ 2>logs/$OMPI_COMM_WORLD_RANK.log
fi
