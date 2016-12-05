import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

#np.set_printoptions(threshold=np.nan)

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print "this is a", np.where(mnist.test.labels[0]==1)[0].squeeze()
matrix = np.reshape(mnist.test.images[0],(28,28))

print matrix

plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
plt.colorbar()
plt.show()



exit()

# define # of input features
N_inputfeats=784
# define output feats (classifier)
N_class=10


# Create the model
x = tf.placeholder(tf.float32, [None, N_inputfeats])
W = tf.Variable(tf.zeros([N_inputfeats, N_class]))
b = tf.Variable(tf.zeros([N_class]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, N_class])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# train net
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

