# Restore the Model

import tensorflow as tf

sess = tf.Session()

saver = tf.train.import_meta_graph('./Model1/my_test_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./Model1/'))

# Accessing the variable named 'bias'
print(sess.run('bias:0')) # value of this variable is 2


# Accessing the Placeholders

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict = {w1:13.0, w2:17.0} # 13.0 + 17.0 = 30.0


# Accessing the operation (op) that we want to run

op_to_restore =  graph.get_tensor_by_name("op_to_restore:0")

print(sess.run(op_to_restore, feed_dict)) # 30.0 * 2 = 60.0
