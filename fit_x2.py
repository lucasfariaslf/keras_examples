
# coding: utf-8

import numpy as np
import matplotlib.pyplot as mpl

# K here calls for TensforFlow, default backend
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input

# simple loss function comparing algebraic and
# automatic differantiation
def custom_loss(input_tensor, output_tensor):
    def loss(y_true, y_pred):
        
        # graph derivative
        gradients = K.gradients(output_tensor, input_tensor)
        grad_pred = K.sum(gradients, axis=-1)
        
        # analytical derivative
        grad_true = K.sum(2*input_tensor, axis=-1)

        return K.square(grad_pred - grad_true)
    return loss

# layers set up
input_tensor = Input(shape=(1,)) # each entry of n_th entry sets size of dimension

# no clear reason when to use relu or sigmoid
hidden_tensor1 = Dense(10, activation='relu')(input_tensor)
hidden_tensor2 = Dense(10, activation='relu')(hidden_tensor1)
hidden_tensor3 = Dense(10, activation='relu')(hidden_tensor2)
output_tensor = Dense(1, activation='softplus')(hidden_tensor3)


model = Model(input_tensor, output_tensor)
model.compile(loss=custom_loss(input_tensor, output_tensor), optimizer='adam')
# model.compile(loss='mean_squared_error', optimizer='adam')

# input data and labels
x = np.linspace(-10, 10, num=2000)
labels = np.array([x[i]**2 for i in range(len(x))])

model.fit(x=x,y=labels, batch_size=10, epochs=1000, verbose=2)  # starts training

# predict using defaults
yhat = model.predict(x)

# Plotting
mpl.scatter(x,labels)
mpl.scatter(x,yhat)
mpl.show()