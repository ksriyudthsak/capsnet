# -*- coding: utf-8 -*-

# 20180401
# #### Capsule Networks (CapsNets) ####
# Content was based on the paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829), by Sara Sabour, Nicholas Frosst and Geoffrey E. Hinton (NIPS 2017) ].
# Codes was modified from https://github.com/ageron/handson-ml for my study and writing a blog
# For more explanation, please see my blog [Capsule network（新 neural network）で毒キノコ画像を判別してみた]

# To support both Python 2 and Python 3:
from __future__ import division, print_function, unicode_literals

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import random

# Set the random seeds so that the calculation always produces the same output
np.random.seed(42)
tf.set_random_seed(42)

# set up parameters
restore_checkpoint = False#True
n_epochs = 1000 #5#100
num_batch_size = 4 #4#10
num_train_samples = 2000 #20#1000
num_validate_samples = 1000 #20#500
num_test_samples = 1000 #20#500
num_label = 4

# set image size
set_size = 28 #128 #92 #28
set_kernel_size = 9 #59 #41 #9
colour_mode = 3 # channels=1 for grayscale, channels=3 for RGB

def input_image(csv_name, num_batch_size):
    # load mushroom data
    fname_queue = tf.train.string_input_producer([csv_name])
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])
    # decode and resize images
    jpeg_r = tf.read_file(fname)
    image = tf.image.decode_jpeg(jpeg_r, channels=colour_mode)
    image = tf.cast(image, tf.float32)
    resized_image = tf.image.resize_images(image, [set_size,set_size])
    # create tensorflow batch
    image_batch, label_batch = tf.train.batch([resized_image, label], batch_size=num_batch_size)
    return image_batch, label_batch

def image_augmentation(x_train, y_train, num_batch_size):
    train_datagen_augmented = ImageDataGenerator(
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=.1,
            horizontal_flip=True,
            vertical_flip=True)
    train_datagen_augmented.fit(x_train)
    x_train = train_datagen_augmented.flow(x_train, y_train, batch_size=num_batch_size)
    return x_train, y_train

# create placeholders for images (X) and labels (y)
X = tf.placeholder(shape=[None, set_size, set_size, colour_mode], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

# # Primary Capsules
# The first layer will be composed of 32 maps of 6×6 capsules each, where each capsule will output an 8D activation vector:
caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6   # 1152 primary capsules
caps1_n_dims = 8

# To compute their outputs, we first apply two regular convolutional layers:
conv1_params = {
    "filters": 256,
    "kernel_size": set_kernel_size,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": set_kernel_size,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
print ("show_caps1_raw", caps1_raw)

# create squash function proposed in the capsnet paper
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

# apply squash function as the output of each primary capsules
caps1_output = squash(caps1_raw, name="caps1_output")
print ("show_caps1_output",caps1_output )


# # Digit Capsules
caps2_n_caps = num_label
caps2_n_dims = 16
init_sigma = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

# create the first array by repeating `W` once per instance:
batch_size = tf.shape(X)[0]
print ("show_batch_size", batch_size)
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")

# check the shape of the first and second arrays
print ("show_W_tiled:",W_tiled)
print ("show_caps1_output_tiled:",caps1_output_tiled)

caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
print ("show_caps2_predicted: ",caps2_predicted)

# ## Routing by agreement
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")
# ### Round 1
# apply the softmax function to compute the routing weights (equation (3) in the paper):
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

# compute the weighted sum of all the predicted output vectors for each second-layer capsule (equation (2)-left in the paper):
weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")
print ("show_caps2_output_round_1", caps2_output_round_1)

# ### Round 2
print ("show_caps2_predicted:\t",caps2_predicted)
print ("show_caps2_output_round_1:\t",caps2_output_round_1)
caps2_output_round_1_tiled = tf.tile(caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled, transpose_a=True, name="agreement")
raw_weights_round_2 = tf.add(raw_weights, agreement, name="raw_weights_round_2")

# The rest of round 2 is the same as in round 1:
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2, dim=2, name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2, caps2_predicted, name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2, axis=1, keep_dims=True, name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2, axis=-2, name="caps2_output_round_2")
caps2_output = caps2_output_round_2

# ### Static or Dynamic Loop?
# For example, here is how to build a small loop that computes the sum of squares from 1 to 100:
def condition(input, counter):
    return tf.less(counter, 100)

def loop_body(input, counter):
    output = tf.add(input, tf.square(counter))
    return output, tf.add(counter, 1)

with tf.name_scope("compute_sum_of_squares"):
    counter = tf.constant(1)
    sum_of_squares = tf.constant(0)

    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])
    

with tf.Session() as sess:
    print(sess.run(result))

print (sum([i**2 for i in range(1, 100 + 1)]))

# # Estimated Class Probabilities (Length)
# The lengths of the output vectors represent the class probabilities, so we could just use `tf.norm()` to compute them, but as we saw when discussing the squash function, it would be risky, so instead let's create our own `safe_norm()` function:
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

# To predict the class of each instance, we can just select the one with the highest estimated probability.
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
print ("show_y_proba_argmax",y_proba_argmax)
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")


# # Margin loss
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")
print ("show_T", T)

# A small example should make it clear what this does:
with tf.Session():
    print(T.eval(feed_dict={y: np.array([0, 1])}))

print ("show_caps2_output",caps2_output)
# The 16D output vectors are in the second to last dimension, so using the `safe_norm()` function with `axis=-2`:
caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, num_label), name="present_error")
print ("show_present_error", present_error)

absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, num_label), name="absent_error")
print ("show_lambda_", lambda_)
print ("show_absent_error", absent_error)
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

# # Reconstruction
# ## Mask
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets, depth=caps2_n_caps, name="reconstruction_mask")
print ("show_reconstruction_mask",reconstruction_mask)
print ("show_caps2_output comparing to reconstruction mask", caps2_output)

reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1], name="reconstruction_mask_reshaped")
caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask_reshaped, name="caps2_output_masked")
print ("show_caps2_output_masked",caps2_output_masked)


# One last reshape operation to flatten the decoder's inputs:
decoder_input = tf.reshape(caps2_output_masked, [-1, caps2_n_caps * caps2_n_dims], name="decoder_input")
print ("show_decoder_input", decoder_input)


# ## Decoder
# decode by two dense (fully connected) ReLU layers followed by a dense output sigmoid layer:
n_hidden1 = 512 * colour_mode
n_hidden2 = 1024 * colour_mode
n_output = set_size * set_size * colour_mode

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name="decoder_output")
print ("show_decoder_output",decoder_output)

# ## Reconstruction Loss
# reconstruction loss = squared difference between the input image and the reconstructed image:
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference, name="reconstruction_loss")


# ## Final Loss
# final loss = sum of the margin loss and the reconstruction loss (scaled down by a factor of 0.0005 to ensure the margin loss dominates training)
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

# ## Accuracy
correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# ## Training Operations
# the paper used the Adam optimizer with TensorFlow's default parameters which is 0.001 but the parameter was adjusted to 0.0001 for this model
optimizer = tf.train.AdamOptimizer(0.0001)
training_op = optimizer.minimize(loss, name="training_op")

# ## Initializer and Saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# # Training
def augmentation_plot(X_batch, num_batch_size, iteration, filename):
    plt.figure()
    n_samples = num_batch_size
    for index in range(n_samples):
        plt.subplot(2, n_samples/2, index + 1)
        plt.title(filename+"_:" + str(iteration))
        plt.imshow(X_batch[index]/255, interpolation='nearest')
        plt.axis("off")
    plt.savefig("output/"+filename+"_"+str(iteration) + ".jpg")

print ("=========== start training the model ===========")
n_iterations_per_epoch = num_train_samples // num_batch_size
n_iterations_validation = num_validate_samples // num_batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"
fout = open("output/train_result.txt", 'w')

train_image_batch, train_label_batch = input_image('downloads/picture_name_train.csv', num_batch_size)
validate_image_batch, validate_label_batch = input_image('downloads/picture_name_validate.csv', num_batch_size)
test_image_batch, test_label_batch = input_image('downloads/picture_name_test.csv', num_batch_size)

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    # initialisation
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess)

    # training and validating loop
    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            # run the training operation and measure the loss:
            X_batch = train_image_batch.eval()
            y_batch = train_label_batch.eval()

            # augment image for every iteration
            X_batch = X_batch / 255
            X_batch1, y_batch = image_augmentation(X_batch, y_batch, num_batch_size)
            if random.uniform(0, 1) < 0.99:
                X_batch = next(X_batch1)
                X_batch = X_batch[0]
            else:
                X_batch = X_batch
            X_batch = X_batch * 255

            # display augmentation image
            if (epoch == 1) and (iteration < 5):
                augmentation_plot(X_batch, num_batch_size, iteration, "augment_train")

            _, loss_train = sess.run([training_op, loss],
                feed_dict={X: X_batch.reshape([-1, set_size, set_size, colour_mode]), y: y_batch})

            print ("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train), end="")

        # measure the validation loss and accuracy at the end of each epoch
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch = validate_image_batch.eval()
            y_batch = validate_label_batch.eval()

            # augment image for every iteration
            X_batch = X_batch / 255
            X_batch1, y_batch = image_augmentation(X_batch, y_batch, num_batch_size)
            if random.uniform(0, 1) < 0.50:
                X_batch = next(X_batch1)
                X_batch = X_batch[0]
            else:
                X_batch = X_batch
            X_batch = X_batch * 255

            # display augmentation image
            if (epoch == 1) and (iteration < 2):
                augmentation_plot(X_batch, num_batch_size, iteration, "augment_validate")
            loss_val, acc_val = sess.run([loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, set_size, set_size, colour_mode]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print ("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val, " (improved)" if loss_val < best_loss_val else ""))
        fout.write("\rEpoch:\t{}\tVal accuracy:\t{:.4f}\t%\tLoss:\t{:.6f}\t{}".format(
            epoch + 1, acc_val * 100, loss_val, " (improved)" ))

        # save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
fout.close()

print ("=========== start testing the model ===========")
n_iterations_test = num_test_samples // num_batch_size
fout = open("output/test_result.txt", 'w')
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    tf.train.start_queue_runners() # restore sess so do not need to pass sess

    # testing loop
    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch = test_image_batch.eval()
        y_batch = test_label_batch.eval()

        # augment image for every iteration
        X_batch = X_batch / 255
        X_batch1, y_batch = image_augmentation(X_batch, y_batch, num_batch_size)
        if random.uniform(0, 1) < 0.50:
            X_batch = next(X_batch1)
            X_batch = X_batch[0]
        else:
            X_batch = X_batch
        X_batch = X_batch * 255

        if (epoch == 1) and (iteration < 2):
            augmentation_plot(X_batch, num_batch_size, iteration, "augment_test")
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, set_size, set_size, colour_mode]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test), end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))
    fout.write("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(acc_test * 100, loss_test))
fout.close()


print ("=========== start predicting image using the trained model ===========")
n_samples = num_batch_size # number of sample selected for the prediction
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    tf.train.start_queue_runners() # restore sess so do not need to pass sess
    X_batch = test_image_batch.eval()
    y_batch = test_label_batch.eval()
    sample_images = X_batch[:n_samples].reshape([-1, set_size, set_size, colour_mode])
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})

print ("=========== start plotting the predicted images ===========")
sample_images = sample_images.reshape(-1, set_size, set_size, colour_mode)
reconstructions = decoder_output_value.reshape([-1, set_size, set_size, colour_mode])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(2, n_samples/2, index + 1)
    plt.imshow(sample_images[index]/255, interpolation='nearest')
    plt.title("Label:" + str(y_batch[index]))
    plt.axis("off")
plt.savefig("output/label.jpg")

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(2, n_samples/2, index + 1)
    plt.imshow(reconstructions[index]/255, interpolation='nearest')
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.axis("off")
plt.savefig("output/prediction.jpg")

