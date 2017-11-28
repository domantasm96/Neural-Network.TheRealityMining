import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math
from sklearn.neural_network import BernoulliRBM

data = scipy.io.loadmat("realitymining.mat")['s']
affilation = data['my_affil']
data_mat = data['data_mat']
sloan_list = []
features_list = []
observation_days = 7
count = 0

sloan_list = []
features_list = []
frequency_list = []
observation_days = 7
count = 0
frequency_labels = []

for i in range(len(affilation[0])):
    if len(data_mat[0][i]) > 0 and len(affilation[0][i]) > 0:
        frequency_list += [data_mat[0][i]]
        if affilation[0][i][0][0][0] == 'sloan' or affilation[0][i][0][0][0] == 'sloan_2':
            frequency_labels += [1]
        else:
            frequency_labels += [0]
        if len(data_mat[0][i][0]) >= observation_days:
            features_list += [data_mat[0][i]]
            if affilation[0][i][0][0][0] == 'sloan' or affilation[0][i][0][0][0] == 'sloan_2':
                sloan_list += [count]
            count += 1

for subject in range(len(features_list)):
    for hour in range(len(features_list[subject])):
        for element in range(len(features_list[subject][hour])):
            if math.isnan(features_list[subject][hour][element]):
                features_list[subject][hour][element] = 4

# 0 – no signal, 1 – home, 2 – work, 3 – elsewhere, 4 – phone is off

all_places = [0, 1, 2, 3, 4]
count = 0

features = np.zeros((len(features_list), observation_days * 24 * len(all_places)))

for subject in range(len(features_list)):
    for number in range(len(all_places)):
        for week in range(observation_days):
            for hours in range(24):
                if features_list[subject][hours][week] == all_places[number]:
                    features[subject][count] = 1
                count += 1
    count = 0

rbm_list = []
sk_rbm = BernoulliRBM(n_components=100, verbose=True, learning_rate=0.1, n_iter=1000)
sk_rbm.fit(features)
rbm_list = sk_rbm.transform(features)

labels = []
for i in range(len(features)):
    if i in sloan_list:
        labels += [1]
    else:
        labels += [0]

rbm_labels = np.array(labels)

# 1 – home, 2 – work, 3 – elsewhere, 0 – no signal, 4 – phone is off
home, work, elsewhere, no_signal, phone_off = 0, 0, 0, 0, 0
frequency = []
temp = []
for subject in range(len(frequency_list)):
    for hours in range(24):
        for elements in range(len(frequency_list[subject][hours])):
            if frequency_list[subject][hours][elements] == 1:
                home += 1
            elif frequency_list[subject][hours][elements] == 2:
                work += 1
            elif frequency_list[subject][hours][elements] == 3:
                elsewhere += 1
            elif frequency_list[subject][hours][elements] == 0:
                no_signal += 1
            else:
                phone_off += 1
        temp += [home / len(frequency_list[subject][hours]) if home != 0 else 0,
                 work / len(frequency_list[subject][hours]) if work != 0 else 0,
                 elsewhere / len(frequency_list[subject][hours]) if elsewhere != 0 else 0,
                 # no_signal/len(frequency_list[subject][hours]) if no_signal !=0 else 0,
                 phone_off / len(frequency_list[subject][hours]) if phone_off != 0 else 0]
        home, work, elsewhere, no_signal, phone_off = 0, 0, 0, 0, 0
    frequency += [temp]
    temp = []
frequency_features = np.array(frequency)

X_train, X_test, y_train, y_test = train_test_split(features, rbm_labels, test_size=0.3, random_state=0)
rbm_list = []
sk_rbm = BernoulliRBM(n_components=100, verbose=True, learning_rate=0.1, n_iter=1000)
sk_rbm.fit(X_train)
X_train = sk_rbm.transform(X_train)

# Parameters that define the MLP
n_inputs = len(X_train[0])
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 2
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        # Number of inputs
        n_inputs = int(X.get_shape()[1])

        # This value is computed to randomly initialize the weights
        stddev = 2 / np.sqrt(n_inputs)
        # Weigths can be initialized in different ways
        # Here they are randomly initialized from a Normal distribution (mean=0,std as computed before)
        # Notice that weights are organized in a matrix (tensor) and its number is n_inputs*n_neurons
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # The variable that will contain the weights is W
        W = tf.Variable(init, name="kernel")

        # The variable that will contain the bias is b  
        # and is initialized to zero
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        # As in the perceptron what the neurons do is multiply the weights by 
        # the input

        Z = tf.matmul(X, W) + b

        # What the activation function does is to "process" the result
        # of the multiplication of weights by inputs, and this is the output
        # of every neuron. 

        if activation is not None:
            return activation(Z)
        else:
            return Z


# The scope name for this MLP is "dnn"
with tf.name_scope("dnn"):
    # The first hidden layer is defined using the RELU activation function
    # It will contain n_hidden1=300 hidden neurons and therefore output
    # 300 values    
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)

    # The second hidden layer is also defined using the RELU activation function
    # It will contain n_hidden2=100 hidden neurons and therefore output
    # 100 values    
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)

    # The output layer does not use any activation function
    # it will output n_outputs=10 values since there are 10 classes in MNIST
    logits = neuron_layer(hidden2, n_outputs, name="outputs")
# scope name of the loss function is "loss"
with tf.name_scope("loss"):
    # The Loss function is defined. It outputs a loss value for each x
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    #     xentropy = tf.contrib.keras.backend.binary_crossentropy(output=n_inputs, target = n_inputs, from_logits=False)

    # The total loss is mean of the loss values 
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    # The plain GradientDescentOptimizer is chosen
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    # A prediction is correct if it is among the k=1 most probable
    # classes predicted by the NN. Since k=1, it is only correct
    # if the prediction coincides with the true class.
    correct = tf.nn.in_top_k(logits, y, 1)

    # The accuracy is the mean of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialization of the computation graph
init = tf.global_variables_initializer()

# tensorflow allows to define a saver to store the model after learning
saver = tf.train.Saver()

# Number of epochs
n_epochs = 100

# Size of the batch used to update the gradient
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(70 // batch_size):     
            #Function next_batch automatically select the batch
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
            
            # Weights are learned using the current batch            
            sess.run(training_op, feed_dict={X: X_train, y: y_train})
            
        # Accuracies are computed in the training and validation sets    
        acc_train = accuracy.eval(feed_dict={X:  X_train, y: y_train})
        acc_val = accuracy.eval(feed_dict={X: sk_rbm.transform(X_test),
                                            y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

# Classificaton from RBM feature
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(frequency_features, frequency_labels, test_size=0.3, random_state=0)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

model = clf.fit(X_train, y_train)
predicted_labels = model.predict(X_test)
print("Accuracy %f" % metrics.accuracy_score(y_test, predicted_labels))
