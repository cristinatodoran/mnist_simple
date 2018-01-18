import tensorflow as tf
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data


class NeuralNet:
    def __init__(self, nClasses, image_size, learningRate):
        self.n_classes = nClasses
        self.image_size = image_size
        self.learningRate = learningRate
        self.X = tf.placeholder(dtype='float', shape=[None, self.image_size], name='X')  # height, width
        self.Y = tf.placeholder(dtype='int32', name='Y')

    def model(self, num_nodehl1, num_nodehl2, num_nodehl3):
        self.n_nodes_hl1 = num_nodehl1
        self.n_nodes_hl2 = num_nodehl2
        self.n_nodes_hl3 = num_nodehl3

        self.hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([self.image_size, self.n_nodes_hl1])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        self.hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        self.hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3]))}

        self.output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                             'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        # X * weights + biases
        self.l1 = tf.add(tf.matmul(self.X, self.hidden_layer_1['weights']), self.hidden_layer_1['biases'])
        self.l1 = tf.nn.relu(self.l1)

        self.l2 = tf.add(tf.matmul(self.l1, self.hidden_layer_2['weights']), self.hidden_layer_2['biases'])
        self.l2 = tf.nn.relu(self.l2)

        self.l3 = tf.add(tf.matmul(self.l2, self.hidden_layer_3['weights']), self.hidden_layer_3['biases'])
        self.l3 = tf.nn.relu(self.l3)

        self.predicted_class = tf.add(tf.matmul(self.l3, self.output_layer['weights']), self.output_layer['biases'])

        return self.predicted_class

mnist = input_data.read_data_sets(train_dir=r'D:\DeepLearning_theory_appl\data\mnist', one_hot=False)

image_size = 28 * 28  # input images are 28 x 28
n_classes = 10  # numbers 0 - 9 in the dataset
learningRate = 1e-3
num_nodehl1 = 700
num_nodehl2 = 700
num_nodehl3 = 700
train_batch_size = 128  # manipulate weights by 128 features at a time
epochs = 50


def train():
    neuralNet_ = NeuralNet(n_classes, image_size, learningRate)
    print("Neural Net initialized")
    predictedClasses = neuralNet_.model(num_nodehl1, num_nodehl2, num_nodehl3)
    """
    cross-entropy is a continuous function that is always positive. 
    If the predicted y equals the true y, cross-entropy equals zero.
    Cross-entropy is used in classification.
    This measures how well the model works on each image individually. 
    The softmax_cross_entropy_with_logits function makes the sum of the inputs equal to 1, normalizing the values

    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictedClasses, labels=neuralNet_.Y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

    with tf.Session() as sess:
        # initialize the weights and biases before optimization starts
        sess.run(tf.global_variables_initializer())
        print("Session started, training begins")

        # train the network
        for iter in range(1, epochs + 1):
            iter_loss = 0
            for _ in range(int(mnist.train.num_examples / train_batch_size)):
                # get the next batch of training examples.
                # x_batch holds a batch of images
                # y_true_batch holds the true labels for x_batch
                x_batch, y_true_batch = mnist.train.next_batch(train_batch_size)

                # fd_train holds the batch in a dictionary with the named placeholders
                # in the tensorflow graph
                fd_train = {neuralNet_.X: x_batch, neuralNet_.Y: y_true_batch}
                _, i_loss = sess.run([optimizer, cost], feed_dict=fd_train)

                iter_loss += i_loss
            print('iter', iter, 'of', epochs, 'loss: ', iter_loss)

        correct = tf.nn.in_top_k(predictedClasses, neuralNet_.Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('\ntraining set accuracy: ',
              accuracy.eval({neuralNet_.X: mnist.train.images, neuralNet_.Y: mnist.train.labels}))
        print('test set accuracy: ', accuracy.eval({neuralNet_.X: mnist.test.images, neuralNet_.Y: mnist.test.labels}))

train()