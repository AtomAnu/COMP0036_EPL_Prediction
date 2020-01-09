from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# f1 score metric for the neural network
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

    # build the neural network
    def generate_model(self): 
        model = Sequential()
        model.add(Dense(16, input_dim=self.size_i, activation='relu'))
        model.add(Dense(32, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(16, activation='sigmoid',
                        kernel_regularizer=regularizers.l1(0.01)))
        model.add(Dense(self.size_o, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer, metrics=['accuracy', f1])
        # show the model built above
        model.summary()
        return model
