import tensorflow as tf

def crop_image(images):
    shape = tf.shape(images)
    size = tf.maximum(shape[0], shape[1])
    return tf.image.resize_image_with_crop_or_pad(images, size, size)


def inference(images, keep_prob, batch_size, image_crop_size, input_channels, num_of_classes, variable_with_weight_decay, variable_on_cpu, activation_summary, log_input, log_feature):
    batch_size = tf.cast(batch_size, tf.int32)
    images = tf.reshape(images, [batch_size, image_crop_size, image_crop_size, input_channels])

    if log_input:
        width = images[0].get_shape()[0].value
        height = images[0].get_shape()[1].value
        img = tf.reshape(images[0], [1, width, height, input_channels])
        tf.summary.image('image', img, 1)


    with tf.variable_scope('conv1') as scope:
        out_channels = 96
        kernel = variable_with_weight_decay('weights', shape=[5,5,input_channels,out_channels], stddev=0.01)
        conv = tf.nn.conv2d(images, kernel, [1,2,2,1], padding='SAME')
        #biases = variable_on_cpu('biases', [out_channels], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv1)

        if log_feature:
            width = conv1[0].get_shape()[0].value
            height = conv1[0].get_shape()[1].value
            imgs = tf.reshape(conv1[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv1.op.name, imgs, 10)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        out_channels2 = 192
        kernel = variable_with_weight_decay('weights', shape=[3,3,out_channels,out_channels2], stddev=0.05)
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
        #biases = variable_on_cpu('biases', [out_channels2], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [out_channels2], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv2)

        if log_feature:
            width = conv2[0].get_shape()[0].value
            height = conv2[0].get_shape()[1].value
            imgs = tf.reshape(conv2[0], [1,width,height,out_channels2])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv2.op.name, imgs, 10)

    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    #pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    pool2 = conv2



    with tf.variable_scope('conv3') as scope:
        out_channels3 = 256
        kernel = variable_with_weight_decay('weights', shape=[3,3,out_channels2,out_channels3], stddev=0.05)
        conv = tf.nn.conv2d(pool2, kernel, [1,2,2,1], padding='SAME')
        #biases = variable_on_cpu('biases', [out_channels3], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [out_channels3], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv3)

        if log_feature:
            width = conv3[0].get_shape()[0].value
            height = conv3[0].get_shape()[1].value
            imgs = tf.reshape(conv3[0], [1,width,height,out_channels3])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv3.op.name, imgs, 10)

    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')


    with tf.variable_scope('local3') as scope:
        size = pool3.get_shape()[1] * pool3.get_shape()[2] * pool3.get_shape()[3]
        reshape = tf.reshape(pool3, tf.stack([-1, size]))
        weights = variable_with_weight_decay('weights', shape=[size, 128], stddev=0.05)
        biases = variable_on_cpu('biases', [128], tf.contrib.keras.initializers.he_normal())
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local3)

    #keep_prob1 = tf.convert_to_tensor(0.7)
    local3_dropout = tf.nn.dropout(local3, keep_prob)


    with tf.variable_scope('local4') as scope:
        weights = variable_with_weight_decay('weights', shape=[128, 64], stddev=0.04)
        #biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [64], tf.contrib.keras.initializers.he_normal())
        local4 = tf.nn.relu(tf.matmul(local3_dropout, weights) + biases, name=scope.name)
        activation_summary(local4)

    local4_dropout = tf.nn.dropout(local4, keep_prob)


    """
    with tf.variable_scope('local5') as scope:
        weights = variable_with_weight_decay('weights', shape=[64, 32], stddev=0.04)
        #biases = variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [32], tf.contrib.keras.initializers.he_normal())
        local5 = tf.nn.relu(tf.matmul(local4_dropout, weights) + biases, name=scope.name)
        activation_summary(local5)

    local5_dropout = tf.nn.dropout(local5, keep_prob)
    """


    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('wegiths', [64, num_of_classes], stddev=0.04)
        biases = variable_on_cpu('biases', [num_of_classes], tf.contrib.keras.initializers.he_normal())
        softmax_linear = tf.add(tf.matmul(local4_dropout, weights), biases, name=scope.name)
        activation_summary(softmax_linear)
        softmax = tf.nn.softmax(softmax_linear, name="softmax")

    return softmax_linear