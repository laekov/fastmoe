#!/bin/bash
if [ ! -z $OMPI_COMM_WORLD_LOCAL_RANK ]
then
	export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
fi

if [ -z $OMPI_COMM_WORLD_RANK ]
then
	RANK=single
else
	RANK=$OMPI_COMM_WORLD_RANK 
fi

mkdir -p logs

SCRIPT_PATH=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH=$SCRIPT_PATH:$SCRIPT_PATH/build/lib.linux-x86_64-3.7:$PYTHONPATH
export LD_LIBRARY_PATH=/home/laekov/.local/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
if [ -z $1 ]
then
	python3 tests/moe_test.py 2>logs/$RANK.log
else
	python3 $@ 2>logs/$RANK.log
fi
