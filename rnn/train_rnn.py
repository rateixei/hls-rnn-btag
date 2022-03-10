import numpy as np
import matplotlib.pyplot as plt
import h5py

from tensorflow.keras.utils import to_categorical

import models

train = True

data_loc = '/gpfs/slac/atlas/fs1/d/rafaeltl/public/ML/L1RNN/datasets_2020_ff/'
file_str = 'Jan06_FlavFix_smear_1_std_xtd_zst.h5'

f5 = h5py.File(data_loc+file_str, 'r')
x_train = np.array( f5['x_train'] )
y_train = to_categorical ( np.array( f5['y_train'] ) )
w_train = np.array( f5['w_train'] )

## The big LSTM model
# model, model_name = models.lstmmodel(15, 6, 120, [50, 10], l1_reg=0, l2_reg=0)

## The big GRU model
model, model_name = models.grumodel(15, 6, 120, [50, 10], l1_reg=0, l2_reg=0)

model_json = model.to_json()
with open(f'keras/model_{model_name}_arch.json', "w") as json_file:
    json_file.write(model_json)
    
# adam = Adam(learning_rate=0.01)
model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model_output = f'keras/model_{model_name}_weights.h5'
if train:
    history = model.fit( x_train , y_train,
            batch_size=2**14,
            # epochs=10,
            epochs=150,
            validation_split=0.1,
            shuffle = True,
            sample_weight= w_train,
            callbacks = [
                EarlyStopping(verbose=True, patience=20, monitor='val_accuracy'),
                ModelCheckpoint(model_output, monitor='val_accuracy', verbose=True, save_best_only=True)
                ],
            verbose=True
            )
    
model.load_weights(model_output)
