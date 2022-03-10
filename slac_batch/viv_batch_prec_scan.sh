#!/bin/bash

# KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/notebooks/keras/

KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/rnn/keras/

#MOD=model_rnn_LSTM.120_Dense.5010_LSTMKernelInit.VarianceScaling_DenseKernelInit.lecun_uniformKRl1.0_KRl2.0_recAct.sigmoid
MOD=model_rnn_GRU.120_Dense.5010_LSTMKernelInit.VarianceScaling_DenseKernelInit.lecun_uniformKRl1.0_KRl2.0_recAct.sigmoid

HERE=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/slac_batch/

#REUSE=5000
STRAT="Resource"
prec4=AP_WRAP

for prec1 in 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46
do
    for prec2 in 4 6 8 10
    do
        for prec3 in AP_TRN # default
        #for prec3 in AP_RND_ZERO # best
        do
            #for prec4 in AP_WRAP # default
            #for prec4 in AP_SAT_SYM # best
            for REUSE in 6 50 100 200 500 1000 2000
            do
                if  (( "$prec2" < "$prec1" ))  ; then
                    echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                    jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
                    job="${HERE}/viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
                    # sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=10g --time=100:00:00 ${job}
                    ${HERE}/viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}

                    break 100

                fi
            done
        done
    done
done

# for prec1 in 16 
# do
#     for prec2 in 6
#     do
#         for prec3 in AP_RND AP_RND_ZERO AP_RND_MIN_INF AP_RND_INF AP_RND_CONV AP_TRN AP_TRN_ZERO
#         do
#             for prec4 in AP_SAT AP_SAT_ZERO AP_SAT_SYM AP_WRAP AP_WRAP_SM
#             do
#                 echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
#                 jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
#                 job="./viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
#                 sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
#             done
#         done
#     done
# done

# REUSE=100

# for prec1 in 2 4 6 8 10 12 14 16 18 20 
# do
#     for prec2 in 1 2 4 6 8 10
#     do
#         for prec3 in AP_TRN # default
#         do
#             for prec4 in AP_WRAP # default
#             do
#                 if  (( "$prec2" < "$prec1" ))  ; then
#                     echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
#                     jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}
#                     job="./viv_submit_script.sh ${KERAS_LOC} ${MOD} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT}"
#                     sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3g --time=2:00:00 ${job}
#                 fi
#             done
#         done
#     done
# done

