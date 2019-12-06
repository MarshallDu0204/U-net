import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as Ige


def writeInfo(text):
    with open("info.txt", "a") as f:
        f.write(str(text) + "\n")


def augment(img):
    i = 1
    while i != 95:
        j = 1
        while j != 95:
            if img[i][j] == 1:
                img[i - 1][j] = 1
                img[i + 1][j] = 1
                img[i][j - 1] = 1
                img[i][j + 1] = 1
            j += 1
        i += 1


def binaryToImg(bin):
    axis = []
    for xAxis in bin:
        element = []
        for yAxis in xAxis:
            temp = []
            if yAxis == 0:
                temp.append(0)
                temp.append(0)
                temp.append(0)
            else:
                temp.append(255)
                temp.append(255)
                temp.append(255)
            temp = np.array(temp, dtype='uint8')
            element.append(temp)
        element = np.array(element)
        axis.append(element)

    axis = np.array(axis)
    return axis


def compressImg(img):
    if len(img[0][0]) == 3:
        sample_image = np.asarray(a=img[:, :, 0], dtype=np.uint8)
        return sample_image
    if len(img[0][0]) == 4:
        newImg = []
        i = 0
        while i != 96:
            tempList = []
            j = 0
            while j != 96:
                tempList.append(255 - img[i][j][3])
                j += 1
            newImg.append(tempList)
            i += 1
        img = np.asarray(a=newImg, dtype=np.uint8)
        return img


def readData_single(path):
    path = path + "/train_set_Unet.tfrecords"

    filename_queue = tf.train.string_input_producer([path], num_epochs=20, shuffle=True)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'img': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['img'], tf.uint8)

    image = tf.reshape(image, [512, 512, 1])

    image = tf.cast(image, tf.float32) * (1. / 255)

    label = tf.decode_raw(features['label'], tf.uint8)

    label = tf.reshape(label, [512, 512])

    return image, label


class Unet:

    def __init__(self):

        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.lamb = tf.placeholder(dtype=tf.float32)

        self.unPooling = []

        self.input_image = None

        self.input_label = None

        self.prediction = None

        self.correct_prediction = None

        self.accurancy = None

        self.loss = None

        self.loss_mean = None

        self.loss_all = None

        self.train_step = None

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2])))
        # initial = tf.truncated_normal(shape,stddev=0.01)
        tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.random_normal(shape=shape, dtype=tf.float32)
        return tf.Variable(initial_value=initial)

    def weight_variable_alter(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.28)
        tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(initial))
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    def deconv(self, x, W, O):
        return tf.nn.conv2d_transpose(value=x, filter=W, output_shape=O, strides=[1, 2, 2, 1], padding='VALID')

    def merge_img(self, convo_layer, unsampling):
        return tf.concat(values=[convo_layer, unsampling], axis=-1)

    def setup_network(self, batch_size, mode):

        self.input_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 512, 512, 1])

        self.input_label = tf.placeholder(dtype=tf.int32, shape=[batch_size, 512, 512])

        # first convolution 512*512*1 -->256*256*32

        with tf.name_scope('first_convolution'):

            # ---------conv1----------

            w_conv = self.weight_variable([3, 3, 1, 32])
            b_conv = self.bias_variable([32])

            img_conv = tf.nn.relu(self.conv2d(self.input_image, w_conv) + b_conv)

            X = img_conv

            # ---------conv2-----------
            w_conv = self.weight_variable([3, 3, 32, 32])
            b_conv = self.bias_variable([32])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            self.unPooling.append(X)

            # ---------maxpool--------

            img_pool = self.max_pooling(img_conv)

            X = img_pool

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        # second convolution 256*256*32 --> 128*128*64

        with tf.name_scope('second_convolution'):

            # ---------conv1----------

            w_conv = self.weight_variable([3, 3, 32, 64])
            b_conv = self.bias_variable([64])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            # ---------conv2-----------
            w_conv = self.weight_variable([3, 3, 64, 64])
            b_conv = self.bias_variable([64])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            self.unPooling.append(X)

            # ---------maxpool--------

            img_pool = self.max_pooling(img_conv)

            X = img_pool

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        # third convolution 128*128*64 -->64*64*128

        with tf.name_scope('third_convolution'):

            # ---------conv1----------

            w_conv = self.weight_variable([3, 3, 64, 128])
            b_conv = self.bias_variable([128])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            # ---------conv2-----------
            w_conv = self.weight_variable([3, 3, 128, 128])
            b_conv = self.bias_variable([128])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            self.unPooling.append(X)

            # ---------maxpool--------

            img_pool = self.max_pooling(img_conv)

            X = img_pool

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        # bottom convolution 64*64*128 --->128*128*128

        with tf.name_scope('bottom_convolution'):

            # ---------conv1----------

            w_conv = self.weight_variable([3, 3, 128, 256])
            b_conv = self.bias_variable([256])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            # ---------conv2-----------
            w_conv = self.weight_variable([3, 3, 256, 256])
            b_conv = self.bias_variable([256])

            img_conv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_conv

            # ---------usample--------
            w_conv = self.weight_variable([2, 2, 128, 256])
            b_conv = self.bias_variable([128])

            img_deconv = tf.nn.relu(self.deconv(img_conv, w_conv, [batch_size, 128, 128, 128]) + b_conv)
            X = img_deconv

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        with tf.name_scope('first_deconvolution'):

            tempMatrix = self.unPooling[2]

            # transfer the matrix

            if mode == 1:
                w_conv = self.weight_variable_alter([128 * 128 * 128, 1])
                b_conv = self.bias_variable([128 * 128 * 128])

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 128 * 128 * 128])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                w_conv = self.weight_variable_alter([128 * 128 * 128, 1])
                b_conv = self.bias_variable([128 * 128 * 128])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 128, 128, 128])

            X = self.merge_img(tempMatrix, X)

            # first deconvolution

            w_conv = self.weight_variable([3, 3, 256, 128])
            b_conv = self.bias_variable([128])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([3, 3, 128, 128])
            b_conv = self.bias_variable([128])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([2, 2, 64, 128])
            b_conv = self.bias_variable([64])

            img_deconv = tf.nn.relu(self.deconv(img_deconv, w_conv, [batch_size, 256, 256, 64]) + b_conv)

            X = img_deconv

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        with tf.name_scope('second_deconvolution'):

            tempMatrix = self.unPooling[1]

            # transfer the matrix

            if mode == 1:
                w_conv = self.weight_variable_alter([256 * 256 * 64, 1])
                b_conv = self.bias_variable([256 * 256 * 64])

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 256 * 256 * 64])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                w_conv = self.weight_variable_alter([256 * 256 * 64, 1])
                b_conv = self.bias_variable([256 * 256 * 64])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 256, 256, 64])

            X = self.merge_img(tempMatrix, X)

            # second deconvolution

            w_conv = self.weight_variable([3, 3, 128, 64])
            b_conv = self.bias_variable([64])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([3, 3, 64, 64])
            b_conv = self.bias_variable([64])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([2, 2, 32, 64])
            b_conv = self.bias_variable([32])

            img_deconv = tf.nn.relu(self.deconv(img_deconv, w_conv, [batch_size, 512, 512, 32]) + b_conv)

            X = img_deconv

            X = tf.nn.dropout(X, keep_prob=self.keep_prob)

        with tf.name_scope('final_layer'):

            tempMatrix = self.unPooling[0]

            # transfer the matrix

            if mode == 1:
                w_conv = self.weight_variable_alter([512 * 512 * 32, 1])
                b_conv = self.bias_variable([512 * 512 * 32])

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 512 * 512 * 32])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                w_conv = self.weight_variable_alter([512 * 512 * 32, 1])
                b_conv = self.bias_variable([512 * 512 * 32])

                tempMatrix = tf.nn.relu(tf.matmul(tempMatrix, w_conv) + b_conv)

                tempMatrix = tf.reshape(tempMatrix, [batch_size, 512, 512, 32])

            X = self.merge_img(tempMatrix, X)

            # final layer

            w_conv = self.weight_variable([3, 3, 64, 32])
            b_conv = self.bias_variable([32])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([3, 3, 32, 32])
            b_conv = self.bias_variable([32])

            img_deconv = tf.nn.relu(self.conv2d(X, w_conv) + b_conv)

            X = img_deconv

            w_conv = self.weight_variable([1, 1, 32, 2])
            b_conv = self.bias_variable([2])

            img_deconv = tf.nn.conv2d(input=X, filter=w_conv, strides=[1, 1, 1, 1], padding='VALID')

            self.prediction = tf.nn.bias_add(img_deconv, b_conv)

        # softmax loss

        with tf.name_scope('softmax'):

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction,
                                                                       name='loss')

            self.loss_mean = tf.reduce_mean(self.loss)

            tf.add_to_collection(name='loss', value=self.loss_mean)

            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        with tf.name_scope('accurancy'):

            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32),
                                               self.input_label)

            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)

            self.accurancy = tf.reduce_mean(self.correct_prediction)

        with tf.name_scope('gradient_descent'):

            self.train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss_all)

    def train(self, batch_size, path):

        ckpt_path = path + "/ckpt-unet/model.ckpt"

        tf.summary.scalar("loss", self.loss_mean)

        tf.summary.scalar('accuracy', self.accurancy)

        merged_summary = tf.summary.merge_all()

        model_dir = path + "/unetData/model"

        tb_dir = path + "/unetData/logs"

        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:

            image, label = readData_single(path)

            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=4,
                                                              capacity=1012, min_after_dequeue=1000)

            label_batch = tf.reshape(label_batch, [batch_size, 512, 512])

            sess.run(tf.global_variables_initializer())

            sess.run(tf.local_variables_initializer())

            summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)

            tf.summary.FileWriter(model_dir, sess.graph)

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(coord=coord)

            try:

                epoch = 1

                while not coord.should_stop():

                    example, label = sess.run([image_batch, label_batch])

                    lo, acc, summary = sess.run([self.loss_mean, self.accurancy, merged_summary], feed_dict={
                        self.input_image: example, self.input_label: label, self.keep_prob: 1.0, self.lamb: 0.004
                    })

                    summary_writer.add_summary(summary, epoch)

                    sess.run([self.train_step], feed_dict={
                        self.input_image: example, self.input_label: label, self.keep_prob: 0.6,
                        self.lamb: 0.004
                    })

                    epoch += 1

                    if epoch % 10 == 0:
                        writeInfo(str(epoch) + " " + str(lo) + " " + str(acc))
                        print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))

                    if epoch % 300 == 0:
                        all_parameters_saver.save(sess=sess, save_path=ckpt_path)


            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')


            finally:
                all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                coord.request_stop()

            coord.join(threads)

            print("done training")

    def estimate(self, batch_size, path):
        imgPath = path + "test.tif"

        img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
        img = cv2.resize(src=img, dsize=(512, 512))

        newImg = []
        i = 0
        while i != batch_size:
            newImg.append(img)
            i += 1
        data = newImg
        data = np.reshape(a=data, newshape=(batch_size,512, 512, 1))

        ckpt_path = path + "/ckpt-unet/model.ckpt"

        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            predict_image = sess.run(
                tf.argmax(input=self.prediction, axis=3),
                feed_dict={
                    self.input_image: data,
                    self.keep_prob: 1.0, self.lamb: 0.004
                }
            )

            predict_image = predict_image[0]

            predict_image = binaryToImg(predict_image)
            predict_image = Ige.fromarray(predict_image, 'RGB')
            predict_image.save('predict_image.jpg')
            predict_image.show()

        print('Done prediction')


def main():
    basePath = "/root"
    unet = Unet()
    unet.setup_network(8, 0)
    unet.train(8, basePath)
    unet.estimate(8,basePath)


main()