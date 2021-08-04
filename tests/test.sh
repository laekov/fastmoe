#!/bin/bash
if [ -z $MASTER_ADDR ]
then
    if [ -z $SLURM_JOB_ID ]
    then
        export MASTER_ADDR=localhost
    else
        export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    fi
fi
if [ -z $MASTER_PORT ]
then
    export MASTER_PORT=12215
fi

if [ ! -z $OMPI_COMM_WORLD_RANK ]
then
    RANK=$OMPI_COMM_WORLD_RANK
    localrank=$OMPI_COMM_WORLD_LOCAL_RANK
elif [ ! -z $SLURM_PROCID ]
then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    localrank=$SLURM_LOCALID
else
    RANK=0
    localrank=0
    WORLD_SIZE=1
fi

export CUDA_VISIBLE_DEVICES=$localrank

exec $@
