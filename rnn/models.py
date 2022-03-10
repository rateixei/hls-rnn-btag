from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LSTM, Masking, Input, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import regularizers

def lstmmodel(max_len, n_var, rec_units, ndense=[10], l1_reg=0,
              l2_reg=0, rec_act='sigmoid', extra_lab='none', rec_kernel_init='VarianceScaling',
             dense_kernel_init='lecun_uniform', domask=False):
    
    rec_layer = 'LSTM'
    
    track_inputs = Input(shape=(max_len, n_var,))
    
    if domask:
        hidden = Masking( mask_value=0, name="masking_1")(track_inputs)
    else:
        hidden = track_inputs
    
    if l1_reg > 1e-6 and l2_reg > 1e-6:
        hidden = LSTM(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                  name = 'lstm1_l1l2')(hidden)
    elif l1_reg > 1e-6:
        hidden = LSTM(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l1(l1 = l1_reg),
                  name = 'lstm1_l1')(hidden)
    elif l2_reg > 1e-6:
        hidden = LSTM(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l2(l2 = l2_reg),
                  name = 'lstm1_l2')(hidden)
    else:
        hidden = LSTM(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  name = 'lstm1')(hidden)

    for ind,nd in enumerate(ndense):
        hidden = Dense(nd, activation='relu', kernel_initializer=dense_kernel_init, name=f'dense_{ind}' )(hidden)
    
    output = Dense(3, activation='softmax', kernel_initializer=dense_kernel_init, name = 'output_softmax')(hidden)
    
    model = Model(inputs=track_inputs, outputs=output)
    
    d_layers = ''.join([ str(dl) for dl in ndense ])
        
    if domask:
        mname  = f'MASKED_rnn_{rec_layer}.{rec_units}_Dense.{d_layers}_'
    else:
        mname  = f'rnn_{rec_layer}.{rec_units}_Dense.{d_layers}_'
    mname += f'LSTMKernelInit.{rec_kernel_init}_DenseKernelInit.{dense_kernel_init}'
    mname += f'KRl1.{l1_reg}_KRl2.{l2_reg}_recAct.{rec_act}' #LSTM kernel regularizer
    
    if 'none' not in extra_lab:
        mname += f'_{extra_lab}'
    
    return model, mname

def grumodel(max_len, n_var, rec_units, ndense=[10], l1_reg=0,
              l2_reg=0, rec_act='sigmoid', extra_lab='none', rec_kernel_init='VarianceScaling',
             dense_kernel_init='lecun_uniform', domask=False):
    
    rec_layer = 'GRU'
    
    track_inputs = Input(shape=(max_len, n_var,))
    
    if domask:
        hidden = Masking( mask_value=0, name="masking_1")(track_inputs)
    else:
        hidden = track_inputs
    

    if l1_reg > 1e-6 and l2_reg > 1e-6:
        hidden = GRU(units=rec_units,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l1_l2(l1 = l1_reg, l2 = l2_reg),
                  name = 'gru_l1l2')(hidden)
    elif l1_reg > 1e-6:
        hidden = GRU(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l1(l1 = l1_reg),
                  name = 'gru_l1')(hidden)
    elif l2_reg > 1e-6:
        hidden = GRU(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  kernel_regularizer = regularizers.l2(l2 = l2_reg),
                  name = 'gru_l2')(hidden)
    else:
        hidden = GRU(units=rec_units,
                  recurrent_activation = rec_act,
                  kernel_initializer = rec_kernel_init, 
                  name = 'gru')(hidden)
            

    for ind,nd in enumerate(ndense):
        hidden = Dense(nd, activation='relu', kernel_initializer=dense_kernel_init, name=f'dense_{ind}' )(hidden)
    
    output = Dense(3, activation='softmax', kernel_initializer=dense_kernel_init, name = 'output_softmax')(hidden)
    
    model = Model(inputs=track_inputs, outputs=output)
    
    d_layers = ''.join([ str(dl) for dl in ndense ])
        
    if domask:
        mname  = f'MASKED_rnn_{rec_layer}.{rec_units}_Dense.{d_layers}_'
    else:
        mname  = f'rnn_{rec_layer}.{rec_units}_Dense.{d_layers}_'
    mname += f'LSTMKernelInit.{rec_kernel_init}_DenseKernelInit.{dense_kernel_init}'
    mname += f'KRl1.{l1_reg}_KRl2.{l2_reg}_recAct.{rec_act}' #LSTM kernel regularizer
    
    if 'none' not in extra_lab:
        mname += f'_{extra_lab}'
    
    return model, mname