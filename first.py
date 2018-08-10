import tensorflow as tf
import numpy as np
from text_extract import actual
train_x,train_y,test_x,test_y = actual()
#degining the nodes in each layer
n_nodes_hl1 = 1500
n_nodes_hl2 = 1000
n_nodes_hl3 = 500
n_classes = 30
batch_size = 10
hm_epochs = 16
x = tf.placeholder('float')
y = tf.placeholder('float')
#defining weights and bias of the hidden layers
hidden_1_layer = {'weight': tf.Variable(tf.random_normal([300, n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes]))}


# defining a neural network and the matrix multiplications inside it
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.sigmoid(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output

#method which triggers the whole process
'''
It covers cleaning data + process this data to the model
My training data is 700 sentences and test input is 41
After removing all the JD which are null and their respective Department I've left with 741 entries
'''
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("output",sess.graph)
        sess.run(init)
        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                #this is for the backpropogation in tensorflow as we have to minize the cost and optimize it
                _,c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print(len(test_x),len(test_y),np.array(test_x[0]).shape,np.array(test_y[0]).shape)
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    writer.close()

if __name__=="__main__":
    train_neural_network(x)
