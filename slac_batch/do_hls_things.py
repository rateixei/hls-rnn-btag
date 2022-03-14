import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import h5py

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json, load_model

import plotting
from sklearn.metrics import accuracy_score

import argparse

parser = argparse.ArgumentParser(description='Do HLS Things')

parser.add_argument('-d', '--data', dest='data', type=str, 
                    default='/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/datasets_2020_ff/Jan06_FlavFix_smear_1_std_xtd_zst.h5',
                    help='data to evaluate')

parser.add_argument('-n', '--neural-network', dest='nn', type=str, required=True,
                    help='neural network to test')

parser.add_argument('-o', '--output-dir', dest='out_dir', type=str, default='./',
                    help='Output location')

parser.add_argument('-p', '--precision', dest='prec', type=str, default='16,6',
                    help='HLS precision')

parser.add_argument('-r', '--reuse', dest='reuse', type=int, default=1,
                    help='HLS precision')

parser.add_argument('--profile', dest='prof', default=False, action='store_true',
                    help='Do profiling')

parser.add_argument('--strategy', dest='strat', default='Latency', 
                    help='Strategy')
parser.add_argument('--vivado', dest='viv', action='store_true', default=False,
                    help='Do vivado things')
parser.add_argument('--lut', dest='lut_prec', type=str, default='none', 
                    help='Change LUT precision')

# parser.add_argument('--scratch', dest='set_scratch', action='store_true', default=False,
#                     help='Send project folder to scratch')

args = parser.parse_args()

mod_name = args.nn.split('/')[-1]
mod_arch = args.nn+'_arch.json'
mod_weig = args.nn+'_weights.h5'

out_loc_name = '/'.join( [args.out_dir, mod_name] )

try:
    os.mkdir( out_loc_name )
except FileExistsError:
    print('Directory already exists', out_loc_name)

try:
    os.mkdir( out_loc_name + '/plots' )
except FileExistsError:
    pass

print('--------- Loading network')
model = None
if '.h5' in args.nn:
    model = load_model(args.nn)
else:
    arch_json = open(mod_arch, 'r').read()
    model = model_from_json(arch_json)
    model.load_weights(mod_weig)

if model is None:
    print("Model is none, exiting")
    sys.exit()

print('--------- Loading data')
x_test = None
y_test = None

if 'npy' in args.data:
    if ',' in args.data:
        x_test = np.load(args.data.split(',')[0])
        y_test = np.load(args.data.split(',')[1])
else:
    f5 = h5py.File(args.data, 'r')
    x_test = np.array( f5['x_test'] )
    y_test = to_categorical ( np.array( f5['y_test'] ) )

print('--------- Keras testing')
y_test_keras = model.predict(x_test, batch_size=2**10)
keras_auc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_test_keras, axis=1))
print(f"Keras-Accuracy: {keras_auc}")

######
plt.figure(figsize=(9,9))
_ = plotting.makeRoc(y_test, y_test_keras)
plt.savefig(out_loc_name+'/plots/keras_perf.pdf')
######

import hls4ml

config = hls4ml.utils.config_from_keras_model(model, granularity='name', 
                                              default_precision=f'ap_fixed<{args.prec}>', 
                                              default_reuse_factor=args.reuse)

#print(config)
#sys.exit()
if "Resource" in  args.strat:
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Trace'] = True
    config['Model']['Strategy'] = 'Resource'

#'table_t': 'ap_fixed<18,8>'
#'lstm1': {'Precision': 'ap_fixed<8,4,AP_TRN,AP_WRAP>'}, 'lstm1_tanh': {'Precision': 'ap_fixed<8,4,AP_TRN,AP_WRAP>', 'ReuseFactor': 6, 'table_size': 1024, 'table_t': 'ap_fixed<18,8>'}
#'gru': {'Precision': 'ap_fixed<8,4,AP_TRN,AP_WRAP>'}, 'gru_tanh': {'Precision': 'ap_fixed<8,4,AP_TRN,AP_WRAP>', 'ReuseFactor': 6, 'table_size': 1024, 'table_t': 'ap_fixed<18,8>'}

if 'none' not in args.lut_prec:
    t_size = int(2**float(args.prec.split(',')[1]))
    for layer in config['LayerName'].keys():
        if 'gru' in layer or 'lstm' in layer:
            # config['LayerName'][layer]['table_t'] = f'ap_fixed<{args.lut_prec}>'
            
            config['LayerName'][layer]['table_size'] = f'{t_size}'

print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")

print("\n-----------------------------------")
print('Starting Convert')
hls_model_name = '_'.join( ['model', args.prec.replace(',', '.'), 'reuse', str(args.reuse), args.strat ] )
proj_loc = out_loc_name + f'/{hls_model_name}/myproject_prj/'
#proj_loc = out_loc_name + f'/{hls_model_name}/myproject_prj/'
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir=proj_loc)
print("-----------------------------------")

print("\n-----------------------------------")
print('Starting Compile')
hls_model.compile()
print('Done Compile')
print("-----------------------------------")

print("\n-----------------------------------")
print('Starting Predict')
x_test_cont = np.ascontiguousarray(x_test[:10000,:,:])
y_test_hls = hls_model.predict(x_test_cont)
hls_auc = accuracy_score(np.argmax(y_test[:10000], axis=1), np.argmax(y_test_hls, axis=1))
print(f"HLS-Accuracy: {hls_auc}")
print('Done HLS predict')
print("-----------------------------------")

######
fig, ax = plt.subplots(figsize=(9, 9))

_ = plotting.makeRoc(y_test, y_test_keras)
_ = plotting.makeRoc(y_test[:10000], y_test_hls, linestyle='--')

from matplotlib.lines import Line2D
lines = [Line2D([0], [0], ls='-'),
         Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend
leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
            loc='lower right', frameon=False)
ax.add_artist(leg)
plt.savefig(out_loc_name+f'/plots/keras_vs_hls_perf_{hls_model_name}.pdf')
######

if args.prof:
    import myprofiling
    myprofiling.numerical(model=model, hls_model=hls_model, X=x_test[:1000])


if args.viv:
    
    try:
        os.mkdir( out_loc_name + '/reports' )
    except FileExistsError:
        pass

    print("\n-----------------------------------")
    print('Loading Vivado 2019.2')
    import os
    os.environ['PATH'] = '/gpfs/slac/atlas/fs1/d/rafaeltl/public/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
    os.environ['LM_LICENSE_FILE'] = '2100@rdlic1:2100@rdlic2:2100@rdlic3'
    hls_model.build(csim=False)
    from contextlib import redirect_stdout
    with open(out_loc_name+f'/reports/{hls_model_name}.txt', 'w') as f:
        with redirect_stdout(f):
            hls4ml.report.read_vivado_report(proj_loc)
    with open(out_loc_name+f'/reports/{hls_model_name}.txt', 'a') as f:
        f.write( f"KERAS_AUC {keras_auc}")
        f.write( "\n" )
        f.write( f"HLS_AUC {hls_auc}" )
        f.write( "\n" )
    print('Done Vivado 2019.2', out_loc_name+f'/reports/{hls_model_name}.txt')
    print("-----------------------------------")
