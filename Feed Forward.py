import tensorflow as tf

#scikit learn

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def get_dataset():
    df = pd.read_csv('Iris.csv')
    feature = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
    target = df[['Species']]

    return feature,target

#initiate layer of neural network

layer = {
    'input' : 4,
    'hidden' : 4,
    'output': 3
}

#initiate weight
weight = {
    'input-hidden': tf.Variable(tf.random_normal([layer['input'], layer['hidden']])) ,
    'hidden-output': tf.Variable(tf.random_normal([layer['hidden'], layer['output']]))
}

bias = {
    'input-hidden': tf.Variable(tf.random_normal([layer['hidden']])) ,
    'hidden-output': tf.Variable(tf.random_normal([layer['output']]))
}

#feed data
feature_input =  tf.placeholder(tf.float32 , [None , layer['input']])
target_input =  tf.placeholder(tf.float32 , [None , layer['output']])

def feed_forward():
    #input to hidden
    y1 = tf.matmul(feature_input, weight['input-hidden'])
    y1 += bias['input-hidden']
    y1 =  tf.sigmoid(y1)

    #hidden to output
    y2 = tf.matmul(y1, weight['hidden-output'])
    y2 += bias['hidden-output']

    return tf.sigmoid(y2)

lr = 0.1
#get error result
target_predict = feed_forward()
loss = tf.reduce_min(.5 *(target_input - target_predict) ** 2)

#update weight
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

feature, target = get_dataset()

MinMaxScaler = MinMaxScaler()
scaler = MinMaxScaler.fit(feature)
feature = scaler.transform(feature)

encoder = OneHotEncoder(sparse=False)
encoder = encoder.fit(target)
target = encoder.transform(target)

feat_train, feat_test, tar_train, tar_test = \
    train_test_split(feature,target, test_size=.2)

#activation tensorflow session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_dict = {
        feature_input: feat_train,
        target_input: tar_train
    }

    epoch = 5000

    for i in range(1, epoch + 1):
        sess.run(train, feed_dict=train_dict)

        if i % 200 == 0:
            loss_result = sess.run(loss, feed_dict = train_dict) 
            print("iter: {} loss: {}".format(i, loss_result))

    match = tf.equal(tf.argmax(target_input, axis = 1), tf.argmax(target_predict, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

    test_dict = {
        feature_input : feat_test,
        target_input : tar_test
    }

    print("accuracy: {}".format(sess.run(accuracy, feed_dict = test_dict)))


