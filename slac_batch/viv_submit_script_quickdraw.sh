#!/bin/bash

KERAS_LOC=$1
MOD=$2
prec1=$3
prec2=$4
prec3=$5
prec4=$6
REUSE=$7
STRAT=$8
OUTD=/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/hls/projects/

HERE=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/slac_batch/

SINGULARITY_IMAGE_PATH=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnn_hls_keras_rnn_mastermerge_Feb14.sif

DATA=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/X_test_format.npy,/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/y_test_format.npy

DO_LUT=
DO_LUT="--lut yes"

SUB="python ${HERE}/do_hls_things.py -d ${DATA} -n ${KERAS_LOC}/${MOD} --output-dir ${OUTD} --prec ${prec1},${prec2},${prec3},${prec4} --reuse ${REUSE} --strategy ${STRAT} --lut yes --vivado"

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} ${SUB}