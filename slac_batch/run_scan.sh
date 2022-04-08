#!/bin/bash

KERAS_LOC=''
MOD=''
DATA=''
LTEST=''
TABLE='--lut'
#old image #SING_IMAGE='/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnn_hls_keras_rnn_mastermerge_Feb14.sif'
SING_IMAGE='/gpfs/slac/atlas/fs1/d/rafaeltl/public/sing/rnn_hls_keras_rnn_staticswitch_Apr5.sif'
VIVADO='true'

tot_width=(8 10 12 14 16 18 20 22 24 26 28 30)
int_width=(4 6 8 10 12)

#tot_width=(26)
#int_width=(6)

usage='Option -a) QuickDraw GRU
Option -b) QuickDraw LSTM
Option -c) TopTagging GRU
Option -d) TopTagging LSTM
Option -e) FTAG GRU
Option -f) FTAG LSTM

Option -n) DO NOT run vivado
Option -t) Use OLD table setup
Option -l) Run local test
Option -h) show help'

OPTIND=1
while getopts 'abcdefhltn' flag; do
    case "${flag}" in
        a) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/
           MOD=Quickdraw5Class1GRU.h5
           DATA='/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/X_test_format.npy,/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/y_test_format.npy'
           ;;
        b) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/
           MOD=Quickdraw5ClassLSTMFinL.h5
           DATA='/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/X_test_format.npy,/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/quickdraw/y_test_format.npy'
           ;;
        c) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/
           MOD=model_toptag_gru.h5
           DATA='/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/x_test.npy,/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/y_test.npy'
           ;;
        d) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/
           MOD=model_toptag_lstm.h5
           DATA='/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/x_test.npy,/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/toptagging/y_test.npy'
           ;;
        e) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/rnn/keras/
           MOD=model_rnn_GRU.120_Dense.5010_LSTMKernelInit.VarianceScaling_DenseKernelInit.lecun_uniformKRl1.0_KRl2.0_recAct.sigmoid
           DATA='/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/datasets_2020_ff/Jan06_FlavFix_smear_1_std_xtd_zst.h5'
           ;;
        f) KERAS_LOC=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/rnn/keras/
           MOD=model_rnn_LSTM.120_Dense.5010_LSTMKernelInit.VarianceScaling_DenseKernelInit.lecun_uniformKRl1.0_KRl2.0_recAct.sigmoid
           DATA='/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/datasets_2020_ff/Jan06_FlavFix_smear_1_std_xtd_zst.h5'
           ;;
        l) LTEST='true'
           ;;
        h) echo "$usage"
           return 0
           ;;
        t) TABLE=''
           ;;
        n) VIVADO=''
           ;;
    esac
done

echo 'Keras files location: ' ${KERAS_LOC}
echo 'Model used: ' ${MOD}

if [ $LTEST ]
then
    echo 'Running a local test...'
fi

HERE=/sdf/home/r/rafaeltl/home/rafaeltl/ML/L1BTag/Mar28/hls-rnn-btag/slac_batch/

STRAT="Resource"
prec4=AP_WRAP

for prec1 in "${tot_width[@]}"
do
    for prec2 in "${int_width[@]}"
    #for prec2 in 12
    do
        for prec3 in AP_TRN # default
        do
            #for REUSE in 1 32 64 128 256 512 1024 2048 
            for REUSE in 32
            do
                if  (( "$prec2" < "$prec1" ))  ; then
                    echo "${prec1},${prec2},${prec3},${prec4},${REUSE}"
                    jname=${MOD}_${prec1}_${prec2}_${prec3}_${prec4}_${STRAT}_${REUSE}

                    if [ $VIVADO ] ; then
                        job="${HERE}/submit_vivado.sh ${SING_IMAGE} ${KERAS_LOC} ${MOD} ${DATA} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT} ${TABLE}"
                    else
                        job="${HERE}/submit.sh ${SING_IMAGE} ${KERAS_LOC} ${MOD} ${DATA} ${prec1} ${prec2} ${prec3} ${prec4} ${REUSE} ${STRAT} ${TABLE}"
                    fi

                    if [ $LTEST ] ; then
                        $job
                        break 100
                    else
                        sbatch --partition=usatlas --job-name=${jname} --output=out/${jname}_o.txt --error=err/${jname}_e.txt --ntasks=1 --cpus-per-task=4 --mem-per-cpu=10g --time=48:00:00 ${job}
                    fi


                fi
            done
        done
    done
done
