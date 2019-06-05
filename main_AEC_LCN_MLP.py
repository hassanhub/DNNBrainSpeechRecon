from __future__ import print_function
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, Activation, concatenate, LocallyConnected2D
from keras.layers import MaxPooling1D, MaxPooling2D, Flatten, Merge, LSTM, noise, Reshape, Add, Lambda
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.utils import np_utils, plot_model
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as BK
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU, ELU
import scipy.io as sio
import numpy as np
import h5py
import IPython
import scipy
import random

#GPU configuration
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)

def corr2(a,b):
    k = np.shape(a)
    H=k[0]
    W=k[1]
    c = np.zeros((H,W))
    d = np.zeros((H,W))
    e = np.zeros((H,W))

    #Calculating mean values
    AM=np.mean(a)
    BM=np.mean(b)  

    #Calculating terms of the formula
    for ii in range(H):
      for jj in range(W):
        c[ii,jj]=(a[ii,jj]-AM)*(b[ii,jj]-BM)
        d[ii,jj]=(a[ii,jj]-AM)**2
        e[ii,jj]=(b[ii,jj]-BM)**2

    #Formula itself
    r = np.sum(c)/float(np.sqrt(np.sum(d)*np.sum(e)))
    return r

def corr2_mse_loss(a,b):
    a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
    b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
    tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
    tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
    tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
    tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
    r = -BK.tf.divide(tmp4,tmp3)
    m=BK.tf.reduce_mean(BK.tf.square(BK.tf.subtract(a, b)))
    rm=BK.tf.add(r,m)
    return rm

####two subjects
print('Loading data and models...')
print('Loading data from subject 123...')
h5f = h5py.File('Data_Decoding_LIJ123_HGaLFP_pitch_Train.h5','r')
f0_train = h5f['f0'][:]
vuv_train = h5f['vuv'][:]
aperiodicity_train = h5f['aperiodicity'][:]
spectrogram_train = h5f['spectrogram'][:]
neu_train_123 = h5f['shft_neural_4d'][:]
h5f.close()

h5f = h5py.File('Data_Decoding_LIJ123_HGaLFP_pitch_Test.h5','r')
f0_val = h5f['f0'][:]
vuv_val = h5f['vuv'][:]
aperiodicity_val = h5f['aperiodicity'][:]
spectrogram_val = h5f['spectrogram'][:]
neu_val_123 = h5f['shft_neural_4d'][:]
h5f.close()

print('Loading data from subject 120...')
h5f = h5py.File('Data_Decoding_LIJ120_HGaLFP_pitch_Train.h5','r')
neu_train_120 = h5f['shft_neural_4d'][:]
h5f.close()

h5f = h5py.File('Data_Decoding_LIJ120_HGaLFP_pitch_Test.h5','r')
neu_val_120 = h5f['shft_neural_4d'][:]
h5f.close()

print('Concatenating neural data...')
index=range(98249,2*98249)
neu_train_120=np.delete(neu_train_120,index,axis=0)
neu_train=np.concatenate((neu_train_120,neu_train_123),axis=2)
neu_val=np.concatenate((neu_val_120,neu_val_123),axis=2)

features_train = np.concatenate((spectrogram_train, aperiodicity_train, f0_train, vuv_train), axis=1)
features_val = np.concatenate((spectrogram_val, aperiodicity_val, f0_val, vuv_val), axis=1)
print('Loading and concatenation done.')

Bottleneck='B256'
print('Coding features...')
Encoder_name='AEC_models/'+Bottleneck+'/Encoder_val.h5'
Decoder_name='AEC_models/'+Bottleneck+'/Decoder_val.h5'
encoder = load_model(Encoder_name, custom_objects={'corr2_mse_loss': corr2_mse_loss})
encoded_train = encoder.predict(features_train)
encoded_val = encoder.predict(features_val)
print('Coding done.')

def save_preds(encoded_preds,loss_history,D_name):
    print('Decoding and saving predicted features...')
    decoder = load_model(D_name, custom_objects={'corr2_mse_loss': corr2_mse_loss})
    decoded_preds = decoder.predict(encoded_preds)
    spec=np.power(decoded_preds[0],10)
    aper=-np.power(10,decoded_preds[1])+1
    f0=np.power(10,decoded_preds[2])-1
    vuv=np.round(decoded_preds[3])
    sio.savemat('Main_models/'+Bottleneck+'/Main_preds_Val_AEC_LCN_MLP_LIJ120_123.mat', mdict={'spectrogram':spec.T, 'aperiodicity':aper.T, 'f0':f0.T, 'vuv':vuv.T, 'loss': loss_history})
    print('Saving done.')

#main network
adam=Adam(lr=.0001)
def build_model(shp_in,shp_out):
    reg=.0005
    inputs = Input(shape=shp_in)
    x = LocallyConnected2D(1, kernel_size=[5, 5], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(inputs)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = LocallyConnected2D(1, kernel_size=[3, 3], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = LocallyConnected2D(2, kernel_size=[1, 1], padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.2)(LeakyReLU(alpha=.25)(BatchNormalization()(x)))
    x = Flatten()(x)

    x_MLP = Flatten()(inputs)
    x_MLP = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))
    x_MLP = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x_MLP)
    x_MLP = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x_MLP)))

    x = concatenate([x,x_MLP], axis=1)

    x = Dense(256,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(128,kernel_initializer='he_normal', kernel_regularizer=l2(reg))(x)
    x = Dropout(.3)(ELU(alpha=1.0)(BatchNormalization()(x)))
    x = Dense(shp_out,kernel_initializer='he_normal')(x)
    coded_preds = Activation('tanh', name='coded_preds')(x)
    model = Model(inputs, coded_preds)
    return model

#Inits
adam=Adam(lr=.0001)
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
num_iter=150
shp_in=neu_val.shape[1:]
shp_out=encoded_val.shape[1]
loss_history=np.empty((num_iter,2), dtype='float32')
#cnt_lr=0

model=build_model(shp_in,shp_out)
model.compile(loss=corr2_mse_loss, optimizer=adam)
filepath='Main_models/'+Bottleneck+'/Model_Best_Val_AEC_LCN_MLP_LIJ120_123.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
    
for j in range(num_iter):
    print('#### Iteration:'+str(j+1)+'/'+str(num_iter))
    history = model.fit(neu_train, encoded_train, epochs=1, batch_size=256, verbose=1, callbacks=callbacks_list,  validation_data=(neu_val, encoded_val))
    loss_history[j,0]=history.history['loss'][0]
    loss_history[j,1]=history.history['val_loss'][0]
    #if i>4 and cnt_lr<2:
    #    if loss_history[i,j-5,1]<loss_history[i,j,1] and loss_history[i,j-5,1]<loss_history[i,j-1,1] and loss_history[i,j-5,1]<loss_history[i,j-2,1] and loss_history[i,j-5,1]<loss_history[i,j-3,1] and loss_history[i,j-5,1]<loss_history[i,j-4,1]:
    #    print("########### Validation loss didn't improve after 5 epochs, lr is divided by 2 ############")
    #    BK.set_value(model.optimizer.lr, .5*BK.get_value(model.optimizer.lr))
    #    cnt_lr+=1
model.save('Main_models/'+Bottleneck+'/Model_Val_AEC_LCN_MLP_LIJ120_123.h5')
#model.load_weights(filepath)
encoded_preds = model.predict(neu_val)
h5f = h5py.File('Main_models/'+Bottleneck+'/Encoded_Val_AEC_LCN_MLP_LIJ120_123.h5','w')
h5f.create_dataset('encoded_preds', data=encoded_preds)
h5f.close()
save_preds(encoded_preds,loss_history,Decoder_name)