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

PYTHON_EXEC=python3
PYTHON_VERSION=$($PYTHON_EXEC --version)
PYTHON_REVISION=${PYTHON_VERSION:7:3}

SCRIPT_PATH=$(dirname $(dirname $(realpath $0)))
source ~/scripts/torch.env
export PYTHONPATH=$SCRIPT_PATH:$SCRIPT_PATH/build/lib.linux-x86_64-$PYTHON_REVISION:$PYTHONPATH

core0=$(expr $OMPI_COMM_WORLD_LOCAL_RANK \* 4)
cores=$core0-$(expr $core0 + 3)

exec numactl -C $cores $PYTHON_EXEC $@ 2>logs/$RANK.log
