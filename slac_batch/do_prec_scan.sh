#!/bin/bash

KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/L1BTag/code/hls/notebooks/keras/
# MOD=model_rnn_LSTM_20_10_nomask_KIvs_KRl2.0.0001_recAct.sigmoid
# MOD=model_rnn_LSTM_64_10_nomask_KIvs_KRl20.0001
MOD=model_rnn_LSTM_128_50_nomask_KIvs_KRl2.0.0001_recAct.sigmoid
REUSE=1

for prec1 in 16 
# for prec1 in 16 18 20 22 24 26 28 30 32 34
do
    for prec2 in 6
    # for prec2 in 6 8 10 12
    do
        # for prec3 in AP_RND AP_RND_ZERO AP_RND_MIN_INF AP_RND_INF AP_RND_CONV AP_TRN AP_TRN_ZERO
        for prec3 in AP_RND
        do
            # for prec4 in AP_SAT AP_SAT_ZERO AP_SAT_SYM AP_WRAP AP_WRAP_SM
            for prec4 in AP_SAT_ZERO
            do
                echo "${prec1},${prec2},${prec3},${prec4}"
                jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}
                job='python do_hls_things.py -n ${KERAS_LOC}/${MOD} --prec "${prec1},${prec2},${prec3},${prec4}" --reuse ${REUSE}'
                sbatch --partition=usatlas --job-name=${jname} --output=${jname}_o.txt --error=${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
            done
        done
    done
done
