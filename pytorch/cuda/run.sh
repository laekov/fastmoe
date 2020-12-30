#!/bin/bash
export PYTHONPATH=$PWD/build/lib.linux-x86_64-3.7
export LD_LIBRARY_PATH=/home/laekov/.local/lib/python3.7/site-packages/torch/lib:$LD_LIBRARY_PATH
if [ -z $1 ]
then
	python moe.py
elif [ .$1 = '.test_all' ]
then
	for bs in 4 16 64 
	do
		for inf in 1024 4096
		do
			for ouf in 1024 4096
			do
				for nexp in 4 16 64
				do
					echo $bs $nexp ${inf}x${ouf} 
					python moe_test.py $bs $inf $ouf $nexp
				done
			done
		done
	done
else
	python $@
fi
