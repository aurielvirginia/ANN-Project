import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt 

def loadData():
    df = pd.read_csv("Milk.csv", index_col = "Month")
    df.index = pd.to_datetime(df.index)
    return df

#Load Data
dataset = loadData()

#Show Real Data
#dataset.plot()
#plt.title("Milk Production")
#plt.show()

inputNeuron = 1
hiddenNeuron = 20
outputNeuron = 20

totalBatch = 2
timeStep = 12

testSize = 12
trainSize = len(dataset) - testSize

trainDataset = dataset.head(trainSize)
testDataset = dataset.tail(testSize)

scaler = MinMaxScaler()
trainNormalize = scaler.fit_transform(trainDataset)
testNormalize = scaler.fit_transform(testDataset)

# Architecture RNN
cell = tf.nn.rnn_cell.BasicRNNCell(hiddenNeuron,activation = tf.nn.relu)

# Combine Hidden Neuron
cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_size = outputNeuron , activation = tf.nn.relu)


x = tf.placeholder(tf.float64, [None, timeStep, inputNeuron])
y = tf.placeholder(tf.float64, [None, timeStep, outputNeuron])

output, state = tf.nn.dynamic_rnn(cell, x , dtype = tf.float64)

# Count Loss
# 1. GradientDescentOptimizer
# 2. AdamOptimizer

error = tf.reduce_mean(0.5 * (y - output) ** 2)
train = tf.train.GradientDescentOptimizer(0.1).minimize(error)

# Object for Save Training Data
saver = tf.train.Saver()

def next_batch(dataset):
    xBtach = np.zeros(shape = (totalBatch, timeStep, inputNeuron))
    yBatch = np.zeros(shape = (totalBatch, timeStep, outputNeuron))

    for i in range(totalBatch):
        index = np.random.randint(0, len(dataset) - timeStep)
        xBtach[i] = dataset[index : index + timeStep]
        yBatch[i] = dataset[index + 1 : index + timeStep + 1 ]

    return xBtach, yBatch

init = tf.global_variables_initializer()

epoch = 3000
# Training Data
with tf.Session() as sess:
    sess.run(init)

    for i in range(1, epoch + 1):
        inBatch, outBatch = next_batch(trainNormalize)

        #Create Diction
        inDict = {
            x: inBatch,
            y: outBatch
        }

        sess.run(train, feed_dict = inDict)

        if i % 500 == 0:
            loss = sess.run(error, feed_dict = inDict)
            print("Itteration: {}, Loss: {}".format(i,loss))

    saver.save(sess, "rnn_training/model.ckpt")

#Testing Data
with tf.Session() as sess:
    saver.restore(sess, "rnn_training/model.ckpt")

    seedDataset = list(testNormalize)

    for i in range(timeStep):
        inBatch = np.array(seedDataset[-timeStep:]).reshape(1, timeStep, inputNeuron)
        inDict = {
            x: inBatch
        }

        y = sess.run(output, feed_dict = inDict)

        seedDataset.append(y[0, -1, 0])

# Data Prediction (0 - 1)
prediction = np.array(seedDataset[-timeStep:]).reshape(1, timeStep)

#Invers Data Prediction (Before MinMaxScaler)
prediction = scaler.inverse_transform(prediction)

# Data to Compare
compare = trainDataset.tail(timeStep)

#Add New Column Named Predict
compare["Predict"] = prediction[0]

#Show Plot
compare.plot()
plt.title("Milk Production")
plt.show()
