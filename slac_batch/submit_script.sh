#!/bin/bash

KERAS_LOC=$1
MOD=$2
prec1=$3
prec2=$4
prec3=$5
prec4=$6
REUSE=$7
STRAT=$8

# SINGULARITY_IMAGE_PATH=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnnhls4mlqkeras_no_hls_tk234.sif
SINGULARITY_IMAGE_PATH=/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnnhls4mlqkeras_vDylan_tf241.sif

SUB="python do_hls_things.py -n ${KERAS_LOC}/${MOD} --prec ${prec1},${prec2},${prec3},${prec4} --reuse ${REUSE} --strategy ${STRAT}"

singularity exec --nv -B /sdf,/gpfs,/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} ${SUB}
