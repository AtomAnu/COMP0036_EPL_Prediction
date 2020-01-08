from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()  # .model
        val_targ = self.validation_data[1]  # .model
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f " % _val_f1)
        return

f1 = Metrics()

class Neural_Network:
    def __init__(self, input_size, output_size):
        self.size_i = input_size
        self.size_o = output_size
        self.learning_rate = 0.001
        self.decay = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.optimizer = Adam(lr=self.learning_rate, beta_1=self.beta_1,
                              beta_2=self.beta_2, decay=self.decay)
        self.model = self.generate_model()

    def generate_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.size_i, activation='relu'))
        model.add(Dense(self.size_o, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer, metrics=['accuracy', f1])
        model.summary()
        return model

