import os
import sys

import numpy as np
import copy

from glob import glob
import argparse

import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Do HLS Things')

parser.add_argument('-d', '--data', dest='data', type=str, 
                    required=True, help='data to evaluate')

args = parser.parse_args()

files = glob(args.data)

base_dict = {
    'name':'',
    'BRAM_18K': -1,
    'DSP48E': -1,
    'FF': -1,
    'LUT': -1,
    'URAM':-1,
    'LAT_min':-1,
    'LAT_max': -1,
    'ap_tot': -1,
    'ap_int': -1,
    'ap_quant': '',
    'ap_over': '',
    'strategy': '',
    'reuse': -1,
    'keras_auc': -1,
    'hls_auc': -1,
    'rel_auc': -1
}

dicts = []

for ff in files:
    print(ff)
    with open(ff, 'r') as hf:
        this_dict = copy.deepcopy(base_dict)
        this_dict['name'] = ff.split('/')[-1].replace('.txt', '')
        
        this_dict['reuse'] = int(this_dict['name'].split('_reuse_')[-1].split('_')[0])
        this_dict['strategy'] = this_dict['name'].split('_reuse_')[-1].split('_')[1]

        modes = this_dict['name'].split('_reuse_')[0].replace('model_', '').split('.')

        this_dict['ap_tot'] = int(modes[0])
        this_dict['ap_int'] = int(modes[1])
        this_dict['ap_quant'] = modes[2]
        this_dict['ap_over'] = modes[3]

        for line in hf:
            if 'function' in line or 'none' in line:
                if '_' in line: continue
                
                sline = line.split()
                sline = [ sl.replace('|', '') for sl in sline ]

                this_dict['LAT_min'] = float(sline[3])
                this_dict['LAT_max'] = float(sline[6])

            if 'Total' in line and this_dict['FF'] == -1:
                sline = line.split()
                sline = [ sl.replace('|', '') for sl in sline ]
                
                this_dict['BRAM_18K']  = int( sline[2] )
                this_dict['DSP48E']    = int( sline[3] )
                this_dict['FF']        = int( sline[4] )
                this_dict['LUT']       = int( sline[5] )
                this_dict['URAM']      = int( sline[6] )
            
            if 'KERAS_AUC' in line:
                sline = line.split()
                this_dict['keras_auc'] = float(sline[1])

            if 'HLS_AUC' in line:
                sline = line.split()
                this_dict['hls_auc'] = float(sline[1])

        this_dict['rel_auc'] = abs( this_dict['keras_auc'] - this_dict['hls_auc'] ) / this_dict['keras_auc']
        dicts.append(this_dict)
    # break

# comb_dicts = copy.deepcopy(base_dict)

# for kk in comb_dicts:
#     comb_dicts[kk] = [ dc[kk] for dc in dicts ]

# print(comb_dicts)

## make first plot

int_6 = [ [], [], [], [] ]
int_8 = [ [], [], [], [] ]
int_10 = [ [], [], [], [] ]

re100_int_6 = [ [], [], [], [] ]
re100_int_8 = [ [], [], [], [] ]
re100_int_10 = [ [], [], [], [] ]

for dc in dicts:

    if 'AP_TRN' not in dc['ap_quant']: continue
    if 'ZERO' in dc['ap_quant']: continue
    if 'AP_WRAP' not in dc['ap_over']: continue

    if dc['reuse'] > 1:
        if dc['ap_int'] == 6:
            print(dc['name'])
            re100_int_6[0].append(dc['ap_tot'])
            re100_int_6[1].append(dc['DSP48E'])
            re100_int_6[2].append(dc['LAT_max'])
            re100_int_6[3].append(dc['rel_auc'])
        if dc['ap_int'] == 8:
            re100_int_8[0].append(dc['ap_tot'])
            re100_int_8[1].append(dc['DSP48E'])
            re100_int_8[2].append(dc['LAT_max'])
            re100_int_8[3].append(dc['rel_auc'])
        if dc['ap_int'] == 10:
            re100_int_10[0].append(dc['ap_tot'])
            re100_int_10[1].append(dc['DSP48E'])
            re100_int_10[2].append(dc['LAT_max'])
            re100_int_10[3].append(dc['rel_auc'])
    else:
        if dc['ap_int'] == 6:
            print(dc['name'])
            int_6[0].append(dc['ap_tot'])
            int_6[1].append(dc['DSP48E'])
            int_6[2].append(dc['LAT_max'])
            int_6[3].append(dc['rel_auc'])
        if dc['ap_int'] == 8:
            int_8[0].append(dc['ap_tot'])
            int_8[1].append(dc['DSP48E'])
            int_8[2].append(dc['LAT_max'])
            int_8[3].append(dc['rel_auc'])
        if dc['ap_int'] == 10:
            int_10[0].append(dc['ap_tot'])
            int_10[1].append(dc['DSP48E'])
            int_10[2].append(dc['LAT_max'])
            int_10[3].append(dc['rel_auc'])

fig, ax = plt.subplots()
ax.plot( int_6[0], int_6[1],'o', label='Integer precision = 6' )
ax.plot( int_8[0], int_8[1],'o', label='Integer precision = 8' )
ax.plot( int_10[0], int_10[1],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('DSP usage')
fig.savefig("plot_dsp.pdf")

fig, ax = plt.subplots()
ax.plot( int_6[0], int_6[2],'o', label='Integer precision = 6' )
ax.plot( int_8[0], int_8[2],'o', label='Integer precision = 8' )
ax.plot( int_10[0], int_10[2],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('Maximum latency (us)')
fig.savefig("plot_lat.pdf")

fig, ax = plt.subplots()
print(int_6[0], int_6[3])
ax.plot( int_6[0], int_6[3],'o', label='Integer precision = 6' )
ax.plot( int_8[0], int_8[3],'o', label='Integer precision = 8' )
ax.plot( int_10[0], int_10[3],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('|AUC(HLS)/AUC(Keras) - 1|')
fig.savefig("plot_auc.pdf")

########
########
########

fig, ax = plt.subplots()
ax.plot( re100_int_6[0], re100_int_6[1],'o', label='Integer precision = 6' )
ax.plot( re100_int_8[0], re100_int_8[1],'o', label='Integer precision = 8' )
ax.plot( re100_int_10[0], re100_int_10[1],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('DSP usage')
fig.savefig("re100_plot_dsp.pdf")

fig, ax = plt.subplots()
ax.plot( re100_int_6[0], re100_int_6[2],'o', label='Integer precision = 6' )
ax.plot( re100_int_8[0], re100_int_8[2],'o', label='Integer precision = 8' )
ax.plot( re100_int_10[0], re100_int_10[2],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('Maximum latency (us)')
fig.savefig("re100_plot_lat.pdf")

fig, ax = plt.subplots()
print(int_6[0], int_6[3])
ax.plot( re100_int_6[0], re100_int_6[3],'o', label='Integer precision = 6' )
ax.plot( re100_int_8[0], re100_int_8[3],'o', label='Integer precision = 8' )
ax.plot( re100_int_10[0], re100_int_10[3],'o', label='Integer precision = 10' )
ax.legend()
ax.set_xlabel('Total precision')
ax.set_ylabel('|AUC(HLS)/AUC(Keras) - 1|')
fig.savefig("re100_plot_auc.pdf")