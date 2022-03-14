#!/bin/bash

KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/

MOD=Quickdraw5Class1GRU.h5

HERE=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/slac_batch/

STRAT="Resource"
prec4=AP_WRAP

for prec1 in 8 10 12 14 16 18 20 22 24 26 28 30 #32 34 36 38 40 42 44 46
do
    #for prec2 in 4 6 8 10
    for prec2 in 12
    do
        for prec3 in AP_TRN # default
        do
            for REUSE in 6 50 100 200 500 1000 2000
            do
                if  (( "$prec2" < "$prec1" ))  ; then
                    echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                    jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
                    job="${HERE}/viv_submit_script_quickdraw.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
                    sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=10g --time=50:00:00 ${job}
                    # ${HERE}/viv_submit_script_quickdraw.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}

                    # break 100

                fi
            done
        done
    done
done
