import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

# A graph element can be a Tensor or an Operation
# tf.Session.run runs Operation or evaluates Tensor in fetches
# tf.reduce_mean is an Operation.
# tf.equal is an Operation.


from tensorflow.examples.tutorials.mnist import input_data

# y_true is ground-truth label but hot-encoded. It is a 2 dimensional array.
# y_true_cls is ground-truth label. It is a 1 dimensional array.
# logits is same as scores. It is a 2 dimensional array.
#  y_pred is predicted label but hot-encoded. It is a 2 dimensional array. y_pred = tf.nn.softmax(logits)
# y_pred_cls is predicted label. It is a 1 dimensional array.


# scores = xW+b = logits
# x = tf.placeholder (tf.float32, [None, 28*28]) # 28 * 28 is also called as flattened image size.
# W = tf.Variable (tf.zeros([28*28, 10]))
# b = tf.Variable (tf.zeros([10]))


# cv2 Implementation
def plot_images_with_true_and_ground_truth_labels(images, y_test, predicted_labels = None):
    # destination = np.empty(np.array([28,28])*images.shape[0])
    # print("destination.shape", destination.shape)
    # for image in destination:
    #     cv2.imshow("image", image.reshape(
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    dest = images[0].reshape((28,28))

    for image in images[1:]:
        dest = np.concatenate((dest,image.reshape(28,28)), axis=1)

    cv2.imshow("Image Montage", dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# matplotlib Implementation
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape((28,28)), cmap='binary')


        # if cls_pred is None:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


data = input_data.read_data_sets("data/MNIST/", one_hot=True)
# print("data", data)
print("-----------------------TRAINING DATA------------------------------------------")
# print("data.train.labels", data.train.labels)
print("len of data.train.labels", len(data.train.labels)) # 55,000
print("ndim of data.train.labels", data.train.labels.ndim) # 2
print("shape of data.train.labels", data.train.labels.shape) # (55000, 10)
print("data.test.labels[0:5, :]", data.train.labels[0:5,:])

print("----------------------TEST DATA----------------------------------------------")
# print("data.test.labels", data.test.labels)
print("len of data.test.labels", len(data.test.labels)) # 10,000
print("ndim of data.test.labels", data.test.labels.ndim) # 2
print("shape of data.train.labels", data.test.labels.shape) # (10000, 10)
print("data.test.labels[0:5, :]", data.test.labels[0:5,:])


print("-------------------VALIDATION DATA------------------------------------------")
# print("data.validation.lables", data.validation.labels)
print("len of data.validation.labels", len(data.validation.labels)) # 5000
print("ndim of data.validation.labels", data.validation.labels.ndim) # 2
print("shape of data.validation.labels", data.validation.labels.shape) # (5000, 10)
print("data.test.labels[0:5, :]", data.validation.labels[0:5,:])


print("------------------TRAINING DATA: Converting One-Hot Encoded to Ground-Truth Label----------")

# print(np.argmax(data.train.labels[0,range(data.train.labels.shape[1])]))
# print(np.array([np.argmax(row) for row in data.train.labels]).shape)
y_train = np.array([np.argmax(row) for row in data.train.labels]) # We just use one-hot encoded labels when it comes to training data. So this step is not required
print("y.shape", y_train.shape)
print("y[0:5]", y_train[0:5])


print("----------------TEST DATA: Converting One-Hot Encoded to Ground-Truth Label--------------")
y_test = np.array([np.argmax(row) for row in data.test.labels]) # Ground-Truth Label
print("y.shape", y_test.shape)
print("y[0:5]", y_test[0:5])

print("----------------MNIST SPECIFIC : Number of classes, Image dimension--------------------------------------")
no_of_classes = 10
w = 28
h = 28
img_shape = (w,h)


print("-----------------------TEST DATA: PLAYING AROUND WITH IMAGES--------------------------------------------------")
images = data.test.images[0:9] # 9 images
print("images.shape", images.shape) # (9, 784)
plot_images_with_true_and_ground_truth_labels(images, y_test[0:9])
plot_images(images=images, cls_true=data.test.labels[0:9])
print("images.shape", images.shape) # (9, 784)



print("------------------------TENSORFLOW MODEL: BEGINS---------------------------------")

print("----------------------PLACEHOLDER VARIABLES--------------------------------------")
x = tf.placeholder(tf.float32, [None, w*h]) # None means it can hold arbitrary number of images each being a VECTOR of length w*h # (None, 784)
print("x.shape", tf.shape(x))
# scores = tf.placeholder(tf.float32, [None, no_of_classes]) # None means it can hold arbitrary number of scores each being a VECTOR of length no_of_classes
# print("scores.shape", tf.shape(scores))
predicted_labels = tf.placeholder(tf.int64, [None]) # None means (here), ONE DIMENSIONAL VECTOR OF ARBITRARY LENGTH
print("predicted_labels.shape", tf.shape(predicted_labels))

''' y is same as y_true. They are one hot encoded ground-truth labels'''
# y = tf.placeholder(tf.int64, [None, no_of_classes]) # None means it can hold arbitrary number of ground-truth labels each being a VECTOR of length no_of_classes. Here its a one-hot encoded VECTOR.
# print("y.shape", tf.shape(y))

y_true = tf.placeholder(tf.float32, [None, no_of_classes])

''' y_true_cls is the ground-truth integral labels '''
y_true_cls = tf.placeholder(tf.int64, [None])


one_hot_encoded_predicted_labels = tf.placeholder(tf.int64, [None, no_of_classes])
print("one_hot_encoded_predicted_labels.shape", tf.shape(one_hot_encoded_predicted_labels))

print("---------------------VARIABLES TO BE OPTIMIZED----------------------------------")
weights = tf.Variable(tf.zeros([w*h, no_of_classes])) # (784, 10)
print("weights.shape", tf.shape(weights))
biases = tf.Variable(tf.zeros([no_of_classes])) # 10
print("biases.shape", tf.shape(biases))


print("-------------------- MODEL -------------------------------")
# Note we are multiplying x with weights and not the other way around
# wx+b in general are also called as LOGITS
scores = tf.matmul(x, weights) + biases # x : [no of images, 784] , weights : [784, no of classes] , biases : [no of classes]  , result : [no of images, no of classes]
#In scores, jth element of of ith row tells the score for class j in the ith image.
#Or how likely the ith input image is to be of the jth class.

y_pred = tf.nn.softmax(scores) # logits or scores when passed to SOFTMAX turns into probablity

y_pred_cls = tf.argmax(y_pred, dimension=1) # Getting the predicted labels for each image



print("------------------ COST FUNCTION TO BE OPTIMIZED --------------------")

''' Data Loss Per Image'''
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_true)

''' Data Loss '''
cost = tf.reduce_mean(cross_entropy)



print("-------------------OPTIMIZATION METHOD---------------------------")

''' Optimizer '''
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost) # Only adding the Optimizer-object to the Tensorflow Graph for later execution

print("------------------PERFORMANCE MEASURE----------------------------")

''' Performance Measure '''
correct_prediction = tf.equal(y_pred_cls, y_true_cls) # y_pred_cls = tf.argmax(y_pred, dimension=1), where y_pred = tf.nn.softmax(scores) and y_true_cls = tf.placeholder(tf.int64, [None])

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # On an average, how many correct predictions were made.
print("accuracy: ", accuracy)

# sess.run(tf.initialize_all_variables())
print("correct_prediction: ", correct_prediction)

print("--------------------------STOCHASTIC GRADIENT DESCENT -------------------------")
batch_size = 100

''' Used for Training '''
def optimize(no_of_iterations):
    for i in range(no_of_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size) # y_true_batch is one hot encoded ground-truth label.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)
        # optimizer is the return value of tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)

y_test = np.array([np.argmax(row) for row in data.test.labels])
y_test_cls = np.array([np.argmax(row) for row in data.test.labels])

# key is a tensor? Yes. Value maybe a Python Scalar, String, List, or numpy array that can be converted to the same dtype as that of tensor
# In addition to the above chceking, if key is a placeholder, the shape of the value will be checked for compatibility with the placeholder
# x is a placeholder of tf.float32 with (None, 784) : data.test.images is a numpy array of shape (10000, 784)
# y_true is a placeholder of tf.float32 with (None, 10) : data.test.labels is a numpy array of shape (10000, 10)
# y_true_cls is a placeholder of tf.int64 with (None):  y_test is also a numpy array of shape (10000,)

print("-------------------------------TENSORFLOW RUN----------------------------------")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict_test = {x: data.test.images, y_true: data.test.labels, y_true_cls: y_test}

def print_accuracy():
    acc = sess.run(accuracy, feed_dict = feed_dict_test) # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) where  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    print("Accuracy: ", acc)

def print_confusion_matrix():

    cls_true = y_test
    cls_pred = sess.run(y_pred_cls, feed_dict = feed_dict_test) # y_pred_cls = tf.argmax(y_pred, dimension=1) where y_pred = tf.nn.softmax(scores) and scores = tf.matmul(x, weights) + biases
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(no_of_classes)
    plt.xticks(tick_marks, range(no_of_classes))
    plt.yticks(tick_marks, range(no_of_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

# correct_prediction is a tensor. correct_prediction = tf.equal(y_pred_cls, y_true_cls)
def plot_example_errors():
    correct, cls_pred = sess.run([correct_prediction, y_pred_cls], feed_dict = feed_dict_test)
    incorrect = (correct == False)
    images = y_test_cls[incorrect]
    print(images.shape)
    cls_pred = cls_pred[incorrect]
    cls_true = y_test_cls[incorrect]
    print("images.shape", images.shape)
    plot_images(images=images[0:9], cls_true= cls_true[0:9], cls_pred=cls_pred[0:9])

def plot_weights():
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax  in enumerate(axes.flat):

        if i < 10:
            image = w[:,i].reshape(img_shape)
            ax.set_xlabel("Weights:"+str(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')


        ax.set_xticks([])
        ax.set_yticks([])


print_accuracy()
# plot_example_errors()
print_confusion_matrix()
optimize(no_of_iterations=1)

print_accuracy()
# plot_example_errors()
plot_weights()
# optimize(no_of_iterations=9)
# print_accuracy()
# plot_example_errors()
# plot_weights()
plt.show()
