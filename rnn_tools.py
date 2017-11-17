import tensorflow as tf
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

# Define tensorflow graph
class rnn_model:
    
    def __init__(self, sess, data, n_hidden=32, n_layers=1, 
                 learning_rate=0.01, cell='basic'):
        
        self.sess = sess
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = data.shape[1]
        self.input_dim = data.shape[2]
        self.output_dim = 1
        self.cell_selection = cell.lower()
        
        # Select RNN cell
        if self.cell_selection == 'lstm':
            self.cell = tf.contrib.rnn.BasicLSTMCell
        elif self.cell_selection == 'gru':
            self.cell = tf.contrib.rnn.GRUCell        
        else:
            self.cell = tf.contrib.rnn.BasicRNNCell
        
        if n_hidden is not None:
            try:
                iterator = iter(n_hidden)
                self.n_hidden = n_hidden
            except TypeError:
                self.n_hidden = [n_hidden]
            if len(self.n_hidden) < n_layers:
                [self.n_hidden.append(self.n_hidden[-1]) 
                 for i in range(n_layers - len(self.n_hidden))]
        else:
            self.n_hidden = 32
        
        self.inputs, self.targets, self.outputs, self.loss = self.create_network()
        
        self.training_op = optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)
        
    def create_network(self):
        
        inputs = tf.placeholder(tf.float32, [None, self.batch_size, self.input_dim])
        targets = tf.placeholder(tf.float32, [None, self.batch_size, self.output_dim])
        # Create layers of RNN
        # Note - add function for LSTM's, GRU's, etc.
        rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(size) for size in self.n_hidden]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        rnn_output, states = tf.nn.dynamic_rnn(multi_rnn_cell, inputs, dtype=tf.float32)
        stacked_rnn_output = tf.reshape(rnn_output, [-1, self.n_hidden[-1]])
        # Dense layer to output connection
        stacked_outputs = tf.layers.dense(stacked_rnn_output, self.output_dim)
        outputs = tf.reshape(stacked_outputs, [-1, self.batch_size, self.output_dim])
        
        # Loss function MSE
        loss = tf.reduce_sum(tf.square(outputs - targets))
        
        return inputs, targets, outputs, loss
    
    def train(self, X, Y):
        _, loss = self.sess.run([self.training_op, self.loss],
                               feed_dict={
                self.inputs: X,
                self.targets: Y
            })
        
        return loss
    
    def predict(self, X):
        pred = self.sess.run([self.outputs],
                            feed_dict={
                self.inputs: X
            })[0]
        return pred
      
def train(model, X, Y, epochs=1000):
    model.sess.run(tf.global_variables_initializer())
    loss = []
    for i in range(epochs):
        loss.append(np.mean(model.train(X, Y)))
    
    return loss
  
def forecast_plot(loss, train, train_index, test, 
                  test_index, forecast, train_fit, **kwargs):
    
    for kw in kwargs:
        print(kw, ":", kwargs[kw])
    
    plt.figure(figsize=(15,10))
    plt.subplot(3, 1, 1)
    plt.plot(loss)
    plt.title("Training Loss")
    plt.yscale('log')
    
    plt.subplot(3, 1, 2)
    plt.plot(train_index, train, label='Train')
    plt.plot(train_index, train_fit, label='Fitted')
    plt.plot(test_index, test, label='Test')
    plt.plot(test_index, forecast, label='Forecast')
    plt.legend(loc='best')
    plt.title("Time Series Data")
    
    # Calculate error
    mse = np.mean(np.power(test[1] - forecast, 2))
    
    plt.subplot(3, 1, 3)
    plt.plot(test_index, test, label='Test')
    plt.plot(test_index, forecast, label='Forecast')
    plt.legend(loc='best')
    plt.title('RNN Forecast vs. Actual (MSE={:3f})'.format(mse))
    plt.show()