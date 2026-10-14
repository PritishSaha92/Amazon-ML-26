#!/bin/bash
# two lanes: lane 0 does even chunks, lane 1 odd chunks (chunks still started in order 0,1,2,...)
cd ${BER_ROOT:?}; L=$1
for k in $(seq $L 2 15); do bash research/scripts_92/sft_r8_chunk.sh $k || echo "chunk $k FAILED $(date +%T)" >> ${BER_SCRATCH:?}/sfloor/test/r8/progress.txt; done
