import tensorflow as tf

w1 = tf.placeholder(tf.float32, name="w1")
w2 = tf.placeholder(tf.float32, name="w2")
b1 = tf.Variable(2.0, name="bias")
feed_dict = {w1:4, w2:8}

# Operations

w3 = tf.add(w1,w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")

# Session

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# save the Model
saver = tf.train.Saver()

sess.run(w4, feed_dict)

# save the Graph

saver.save(sess, './Model1/my_test_model', global_step=1000)
