import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from scipy.stats import uniform, truncnorm, randint
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random

print(tf.__version__)
import argparse

parser = argparse.ArgumentParser(description='read in options')
parser.add_argument('--prec', dest='prec', help='precision in ap_fixed<>')
args = parser.parse_args()

#prec = '16,6,AP_TRN,AP_SAT'
#prec = '16,6,AP_TRN'
prec = args.prec
#prec = '16,6'
#prec = '32,12'
#prec = '64,24'
#modeltype = 'GRU_l2_0p01_actl2_0p01'
#modeltype = 'LSTM'
modeltype = 'GRU'
#modeltype = 'LSTM_Test'
#model_load = keras.models.load_model('./Quickdraw5Class1GRU.h5')
#model_load = keras.models.load_model('./Quickdraw5Class1LSTM_l1_0p001_l2_0p0001.h5')
#model_load = keras.models.load_model('./Quickdraw5Class1GRU_l1_0p001_l2_0p0001.h5')
model_load = keras.models.load_model('./Quickdraw5Class1GRU.h5')
#model_load = keras.models.load_model('./Quickdraw5Class1.h5')
#model_load = keras.models.load_model('./Quickdraw5Class1LSTM_l2_0p01.h5')
print(model_load.summary())

#Inputs = keras.Input(shape = (100,3))
##x = keras.layers.GRU(1, return_sequences=True, name='gru_13')(Inputs)
#x = keras.layers.LSTM(2, return_sequences=True, name='gru_13')(Inputs)
#x = keras.layers.Flatten()(x)
#x = keras.layers.Dense(32, activation= 'relu')(x)
#x = keras.layers.Dropout(rate = 0.5)(x)
#x = keras.layers.Dense(16, activation = 'relu')(x)
#predictions = keras.layers.Dense(5, activation='softmax', kernel_initializer='lecun_uniform', name='rnn_densef')(x)
#model_quickdraw = keras.Model(inputs=Inputs, outputs=predictions)
model_quickdraw = model_load

X_load = np.load('./X_test.npy',allow_pickle=True)
y_test = np.load('./y_test.npy',allow_pickle=True)
#print('X_load',X_load[0])
#print('y_test',y_test[0])
#X_test = np.ascontiguousarray(X_test)
print("-----------------------------------")
X_test = []
IX = 0
print('starting data manipulation')
for x1 in X_load:
    ix = 0
    tmp = []
    for x2 in x1:
        ix = ix + 1
        tmp.append([x3 for x3 in x2])
        if (ix == 100):
            break
    for x2 in range(ix,100):
        tmp.append([0,0,0])
    X_test.append(tmp)
    IX = IX + 1
    #if (IX>10000): break
print(len(X_test), len(X_test[-2]), len(X_test[-1]))
del X_load
print('done with data manipulation')
X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='int16')
#print(X_test, type(X_test), X_test.shape)

nsamples = 100000
oddmask = np.array([i%20==1 for i in range(len(y_test))],dtype='bool')
#oddmask = np.ones(len(y_test),dtype='bool')
X_test = X_test[oddmask]
y_test = y_test[oddmask]
X_test = X_test[:nsamples,:,:]
y_test = y_test[:nsamples]
print(type(X_test), X_test.shape)
print(y_test[-10:],y_test.shape)
y_keras = model_quickdraw.predict(X_test)
print('y_keras',y_keras[0])

import hls4ml
import plotting
config = hls4ml.utils.config_from_keras_model(model_quickdraw, granularity='name', default_precision='ap_fixed<%s>'%prec, default_reuse_factor=128)
#for layer in config['LayerName'].keys():
#    config['LayerName'][layer]['Trace'] = True
#config['Model']['Strategy'] = 'Resource'

#config['LayerName']['lstm_2_tanh']['table_t'] = 'ap_fixed<%s>'%prec
#config['LayerName']['gru_tanh']['table_t'] = 'ap_fixed<%s>'%prec
print("-----------------------------------")
print("Configuration")
plotting.print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model_quickdraw,
                                                       hls_config=config,
                                                       output_dir='model_1_%s/hls4ml_prj'%modeltype)
                                                       #fpga_part='xcu250-figd2104-2L-e')
#print("-----------------------------------")
#print('Profiling')
#pscan = hls4ml.model.profiling.numerical(model=model_quickdraw, hls_model=hls_model, X=X_test[:1000])
#print(pscan)
#print([type(p) for p in pscan])
#for ip,p in enumerate(pscan):
#    if p is not None:
#      p.savefig('dist%i_%s%s.pdf'%(ip,prec.replace(',','_'),modeltype))
#print("-----------------------------------")
#print('Compare')
#comp = hls4ml.model.profiling.compare(keras_model=model_quickdraw, hls_model=hls_model, X=X_test[:1000])
#comp.savefig('comp%i_%s%s.pdf'%(ip,prec.replace(',','_'),modeltype))
#print("-----------------------------------")
#hls4ml.utils.plot_model(hls_model, show_shapes=False, show_precision=False, to_file='')

print('Starting Compile')
hls_model.compile()
print('Done Compile')

y_hls = hls_model.predict(X_test)
print('Done HLS predict')
print('y_hls',y_hls[0])

y_test = np.eye(5)[y_test]
print("-----------------------------------")
print("Keras  Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

theclasses = ['ant','bee','butterfly','mosquito','snail']

fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.makeRoc(y_test, y_keras, theclasses)
plt.gca().set_prop_cycle(None) # reset the colors
_ = plotting.makeRoc(y_test, y_hls, theclasses, linestyle='--')

from matplotlib.lines import Line2D
lines = [Line2D([0], [0], ls='-'),
         Line2D([0], [0], ls='--')]
from matplotlib.legend import Legend
leg = Legend(ax, lines, labels=['keras', 'hls4ml'],
            loc='lower right', frameon=False)
ax.add_artist(leg)
plt.savefig('roc_%s%s.pdf'%(prec.replace(',','_'),modeltype))