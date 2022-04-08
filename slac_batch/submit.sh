#!/bin/bash

SING_IMAGE=$1
KERAS_LOC=$2
MOD=$3
DATA=$4
prec1=$5
prec2=$6
prec3=$7
prec4=$8
REUSE=$9
STRAT=${10}
TABLE=${11}

OUTD=/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/hls/projects/

HERE=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/slac_batch/

SUB="python ${HERE}/do_hls_things.py -d ${DATA} -n ${KERAS_LOC}/${MOD} --output-dir ${OUTD} --prec ${prec1},${prec2},${prec3},${prec4} --reuse ${REUSE} --strategy ${STRAT} ${TABLE} "

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SING_IMAGE} ${SUB}
