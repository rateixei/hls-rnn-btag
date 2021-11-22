#!/bin/bash

KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/L1BTag/code/hls/notebooks/keras/
MOD=model_rnn_LSTM_20_10_nomask_KIvs_KRl2.0.0001_recAct.sigmoid
# MOD=model_rnn_LSTM_64_10_nomask_KIvs_KRl20.0001
# MOD=model_rnn_LSTM_128_50_nomask_KIvs_KRl2.0.0001_recAct.sigmoid
# MOD=model_rnn_LSTM_128_50_nomask_KIvs_KRl2.0_recAct.sigmoid
# MOD=model_rnn_LSTM_50_101010_nomask_KIvs_KRl2.0_recAct.sigmoid
# MOD=model_rnn_LSTM_50_101010_nomask_KIvs_KRl2.0_recAct.sigmoid_quick # no difference
# MOD=model_rnn_LSTM_10_10_nomask_KIvs_KRl2.0_recAct.sigmoid # no difference
# MOD=model_rnn_LSTM_1_10_nomask_KIvs_KRl2.0_recAct.sigmoid
# MOD=model_rnn_LSTM_10_10_nomask_KIgo_KRl2.0_recAct.sigmoid

REUSE=1
STRAT="Latency"

for prec1 in 2 4 6 8 10 12 14 16 18 20 
do
    for prec2 in 1 2 4 6 8 10
    do
        for prec3 in AP_TRN # default
        do
            for prec4 in AP_WRAP # default
            do
                if  (( "$prec2" < "$prec1" ))  ; then
                    echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                    jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
                    job="./viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
                    sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
                fi
            done
        done
    done
done

for prec1 in 16 
do
    for prec2 in 6
    do
        for prec3 in AP_RND AP_RND_ZERO AP_RND_MIN_INF AP_RND_INF AP_RND_CONV AP_TRN AP_TRN_ZERO
        do
            for prec4 in AP_SAT AP_SAT_ZERO AP_SAT_SYM AP_WRAP AP_WRAP_SM
            do
                echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
                job="./viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
                sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
            done
        done
    done
done

REUSE=100

for prec1 in 2 4 6 8 10 12 14 16 18 20 
do
    for prec2 in 1 2 4 6 8 10
    do
        for prec3 in AP_TRN # default
        do
            for prec4 in AP_WRAP # default
            do
                if  (( "$prec2" < "$prec1" ))  ; then
                    echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                    jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
                    job="./viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
                    sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
                fi
            done
        done
    done
done

