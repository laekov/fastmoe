#!/bin/bash
if [ ! -z $OMPI_COMM_WORLD_LOCAL_RANK ]
then
	export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
fi

if [ -z $MASTER_PORT ]
then
	export MASTER_ADDR=localhost
	export MASTER_PORT=36666
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

exec python3 $@ 2>logs/$RANK.log
