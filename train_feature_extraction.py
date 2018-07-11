import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
training_file = "train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X, y = train['features'], train['labels']

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# TODO: Define placeholders and resize operation.
nb_classes = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# fc8, 1000
fc8W = tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)
# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot = tf.one_hot(y, nb_classes)

cross = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=logits)
loss = tf.reduce_mean(cross)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_operation = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, dropout:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
# TODO: Train and evaluate the feature extraction model.
EPOCHS = 1
batch_size = 128
n_batch=X_train[0]//batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    print("Training...")
    for j in range(EPOCHS):
        X_train,y_train = shuffle(X_train, y_train)
        offset = 0
        for i in range(n_batch):
            #print(offset)
            x_batch = X_train[offset:offset+batch_size]
            y_batch = y_train[offset:offset+batch_size]
            _,train_loss = sess.run([training_operation, loss], feed_dict = {x: x_batch, y:y_batch, dropout:0.8})
            offset = offset + batch_size

        train_accur = evaluate(X_train, y_train)
        valid_accur = evaluate(X_valid, y_valid)
        print("Epochs {}...".format(j+1))
        print("Train accuracy: {:.3f}".format(train_accur))
        print("Validation accuracy: {:.3f}".format(valid_accur))

