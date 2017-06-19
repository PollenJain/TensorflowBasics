import tensorflow as tf
import numpy as np

''' Input data details '''
w = 28
h = 28
no_of_classes = 10


''' Tensorflow Model '''

# Placeholders (Not used in the code) : Specifying shape of a placeholder is optional
X = tf.placeholder(tf.float32, [None, w*h], name="X") # No memory is allocated for the placeholder here
Y = tf.placeholder(tf.float32, [None, no_of_classes], name="Y")  # No memory is allocated for the placeholder here

# Variables (Not used in the code)
w = tf.Variable(tf.zeros([w*h, no_of_classes]), name="w")
b = tf.Variable(tf.zeros([no_of_classes]), name="b")


# More Variables initial values given using numpy array (Not used in the code)
w1 = tf.Variable(np.arange(2*3).reshape(2,3))
b1 = tf.Variable(np.arange(no_of_classes), name="b1")


# Operations and operands
a = tf.placeholder(tf.int32, name="a")  # No memory is allocated for the placeholder here
b = tf.placeholder(tf.int32, name="b")  # No memory is allocated for the placeholder here

s = tf.add(a,b)
feed_dict = {a:10, b:30}


# Initializing Variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())


''' Saving the Tensorflow Model '''

saver = tf.train.Saver()



# Run the operation by feeding the dict
sess.run(s, feed_dict) # Placeholder gets initialized now, memory is allocated now
# print("sum: ", s.eval())

''' Saving the Tensorflow Graph '''
# add_graph is the name given by us
saver.save(sess, './Model/add_graph', global_step=1000)
