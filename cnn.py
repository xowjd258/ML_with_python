from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import os
import re
from PIL import Image
import importlib



TOWER_NAME = 'tower'


# About Model
MOVING_AVERAGE_DECAY = 0.9999
'''
NUM_EPOCHS_PER_DECAY = 1000.0
LEARNING_RATE_DECAY_FACTOR = 0.005
'''
NUM_EPOCHS_PER_DECAY = 10.0
LEARNING_RATE_DECAY_FACTOR = 0.96


# About Input
IMAGE_FILE_EXTS = ['.jpg', '.png', '.jpeg']


class CNN(object):
    MOVING_AVERAGE_DECAY = 0.9999

    def __init__(self, network, num_of_classes, num_of_examples, image_size, image_crop_size=None, log_input=False, to_grayscale=True, log_feature=False, use_fp16=False):
        if image_crop_size == None:
            image_crop_size = image_size

        if image_crop_size > image_size:
            raise ValueError('crop size have to be smaller than image size.')

        self._num_of_classes = num_of_classes
        self._num_of_examples = num_of_examples
        self._image_size = image_size
        self._image_crop_size = image_crop_size
        self._log_input = log_input
        self._log_feature = log_feature
        self._to_grayscale = to_grayscale
        self._input_channels = 1 if to_grayscale == True else 3
        self._use_fp16 = use_fp16
        self._network = importlib.import_module(network, package=None)



    def train(self, total_loss, global_step, batch_size, init_learning_rate = 0.05, num_epochs_per_decay = 10.0, learning_rate_decay_factor = 0.8):
        num_batches_per_epoch = self._num_of_examples / batch_size
        decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

        lr = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        print ('==> Decay_steps: {}, Lr: {}, samples: {}'.format(decay_steps, lr, self._num_of_examples))

        loss_averages_op = self._add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())


        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op



    #++++++++++++++++++++++++++++++++
    ############# NETWORK #############
    #++++++++++++++++++++++++++++++++
    def inference(self, images, keep_prob, batch_size):
        return self._network.inference(images, keep_prob, batch_size, self._image_crop_size, self._input_channels, self._num_of_classes, self._variable_with_weight_decay, self._variable_on_cpu, self._activation_summary, self._log_input, self._log_feature)



    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        logits = tf.cast(logits, tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy



    def _add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op




    def inputs(self, data_dir, batch_size, min_queue_examples, scaling, distorted):
        if distorted == True:
            images, labels, filenames = self._distorted_inputs(data_dir=data_dir, batch_size=batch_size, min_queue_examples=min_queue_examples, scaling=scaling)
        else:
            images, labels, filenames = self._inputs(data_dir=data_dir, batch_size=batch_size, min_queue_examples=min_queue_examples, scaling=scaling)
        
        if self._use_fp16:
            images = tf.cast(images, tf.float16)

        return images, labels, filenames




    def input(self, image, scaling, distorted):
        """
        file_data = tf.read_file(file_path)
        image = tf.image.decode_jpeg(file_data, channels=3)
        """

        image = self._network.crop_image(image)

        if distorted == True:
            height = self._image_size
            width = self._image_size

            new_size = tf.constant([height, width])
            if scaling:
                distorted_image = tf.image.resize_images(image, new_size)

            height = self._image_crop_size
            width = self._image_crop_size
            image = tf.random_crop(distorted_image, [height, width, 3])
            image = tf.image.random_flip_left_right(distorted_image)
            image = tf.image.random_brightness(distorted_image, max_delta=63)
            image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        else:
            height = self._image_crop_size
            width = self._image_crop_size

            new_size = tf.constant([height, width])
            if scaling:
                image = tf.image.resize_images(image, new_size)

            image = tf.image.resize_image_with_crop_or_pad(image, height, width)


        if self._to_grayscale:
            image = tf.image.rgb_to_grayscale(image)

        image = tf.image.per_image_standardization(image)
        image.set_shape([height, width, self._input_channels])

        if self._use_fp16:
            image = tf.cast(image, tf.float16)

        return image




    #++++++++++++++++++++++++++++++++
    ############# INPUT #############
    #++++++++++++++++++++++++++++++++
    def _read_sample(self, input_queue):
        class Record(object):
            pass

        file_data = tf.read_file(input_queue[0])
        #ext = os.path.splitext(input_queue[0].value)[-1]
        #if ext == '.png':
        #   image = tf.image.decode_png(file_data, channels=3)
        #elif ext == '.jpg' or ext == 'jpeg':
        image = tf.image.decode_jpeg(file_data, channels=3)
        image = self._network.crop_image(image)
        #else:
        #   raise ValueError('Cannot Decode image file')
        label = input_queue[1]

        result = Record()
        result.uint8image = image
        result.label = tf.cast([label], tf.int32)
        result.filename = input_queue[0]

        return result;



    def _get_files(self, dir_name):
        filenames = []
        labels = []

        for (path, dir, files) in os.walk(dir_name):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in IMAGE_FILE_EXTS:
                    filenames.append(os.path.join(dir_name, path, filename))
                    labels.append(int(os.path.basename(path)))

        if len(filenames) <= 0:
            raise ValueError('Data File not exists.', 'DataDir: %s' % dir_name)

        return filenames, labels




    def _distorted_inputs(self, data_dir, batch_size, min_queue_examples, scaling):
        filenames, labels = self._get_files(data_dir)

        input_queue = tf.train.slice_input_producer([filenames, labels], shuffle=True)
        sample = self._read_sample(input_queue)

        distorted_image = tf.cast(sample.uint8image, tf.float32)

        height = self._image_size
        width = self._image_size

        new_size = tf.constant([height, width])
        if scaling:
            distorted_image = tf.image.resize_images(distorted_image, new_size)
        height = self._image_crop_size
        width = self._image_crop_size
        distorted_image = tf.random_crop(distorted_image, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=15)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.8, upper=1.2)
        if self._to_grayscale:
            distorted_image = tf.image.rgb_to_grayscale(distorted_image)

        float_image = tf.image.per_image_standardization(distorted_image)

        float_image.set_shape([height, width, self._input_channels])
        sample.label.set_shape([1])

        print('Filling queue with %d samples before starting to train. '
            'This will take a few minutes.' % min_queue_examples)

        return self._generate_image_and_label_batch(float_image, sample.label, sample.filename, min_queue_examples, batch_size, shuffle=True)





    def _inputs(self, data_dir, batch_size, min_queue_examples, scaling):
        num_examples_per_epoch = self._num_of_examples

        filenames, labels = self._get_files(data_dir)

        input_queue = tf.train.slice_input_producer([filenames, labels], shuffle=False)
        sample = self._read_sample(input_queue)

        reshaped_image = tf.cast(sample.uint8image, tf.float32)

        height = self._image_crop_size
        width = self._image_crop_size

        new_size = tf.constant([height, width])
        if scaling:
            reshaped_image = tf.image.resize_images(reshaped_image, new_size)
        reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
        if self._to_grayscale:
            reshaped_image = tf.image.rgb_to_grayscale(reshaped_image)

        float_image = tf.image.per_image_standardization(reshaped_image)

        float_image.set_shape([height, width, self._input_channels])
        sample.label.set_shape([1])


        return self._generate_image_and_label_batch(float_image, sample.label, sample.filename, min_queue_examples, batch_size, shuffle=False)





    def _generate_image_and_label_batch(self, image, label, filename, min_queue_examples, batch_size, shuffle):
        num_preprocess_threads = 16

        if shuffle:
            images, label_batch, filenames = tf.train.shuffle_batch(
                [image, label, filename],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch, filenames = tf.train.batch(
                [image, label, filename],
                batch_size=batch_size,
                num_threads=1,
                capacity=min_queue_examples + 3 * batch_size)

        return images, tf.reshape(label_batch, [batch_size]), filenames




    
    #++++++++++++++++++++++++++++++++++
    ############# GENERAL #############
    #++++++++++++++++++++++++++++++++++
    def _activation_summary(self, x):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))




    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float16 if self._use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var




    def _variable_with_weight_decay(self, name, shape, stddev, wd=None):
        dtype = tf.float16 if self._use_fp16 else tf.float32
        var = self._variable_on_cpu(name, shape, tf.contrib.keras.initializers.he_normal())
        #var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    
    

    #++++++++++++++++++++++++++++++++++
    ###### GENERAL STATIC MEMBER ######
    #++++++++++++++++++++++++++++++++++
    @staticmethod
    def convert_img_to_square_with_pad(root_dir, out_dir):
        print(root_dir, out_dir)
        convert_count = 0
        for (path, dir, files) in os.walk(root_dir):
            target_dir =  path.replace(root_dir, out_dir, 1)
            print(target_dir)
            if os.path.exists(target_dir) == False:
                os.makedirs(target_dir)

            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in IMAGE_FILE_EXTS:
                    img_path = os.path.join(path, filename)
                    out_path = os.path.join(target_dir, filename)

                    CNN._img_to_square_with_pad(img_path, out_path)
                    print('{} => {}'.format(img_path, out_path))
                    convert_count += 1

        return convert_count


    @staticmethod
    def _img_to_square_with_pad(src_file_path, dest_path):
        with Image.open(src_file_path) as img:
            longer_side = max(img.size)
            horizontal_padding = (longer_side - img.size[0]) / 2
            vertical_padding = (longer_side - img.size[1]) / 2
            with img.crop((-horizontal_padding, -vertical_padding, img.size[0] + horizontal_padding, img.size[1] + vertical_padding)) as square_img:
                square_img.save(dest_path)