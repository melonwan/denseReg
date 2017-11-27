# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Additionally it maintains a collection with update_ops that need to be
   updated after the ops have been computed, for example to update moving means
   and moving variances of batch_norm.

   Ops that have different behavior during training or eval have an is_training
   parameter. Additionally Ops that contain variables.variable have a trainable
   parameter, which control if the ops variables are trainable or not.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.python.training import moving_averages

from network.slim import losses
from network.slim import scopes
from network.slim import variables

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'


# the batch_norm here is batch_renorm implementation instead of batch norm
@scopes.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation=None,
               is_training=True,
               trainable=True,
               restore=True,
               scope=None,
               reuse=None):
  """Adds a Batch ReNormalization layer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    center: If True, subtract beta. If False, beta is not created and ignored.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    activation: activation function.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
    a tensor representing the output of the operation.

  """
  inputs_shape = inputs.get_shape()
  with tf.variable_scope(scope, 'BatchReNorm', [inputs], reuse=reuse):
    axis = list(range(len(inputs_shape) - 1))
    params_shape = inputs_shape[-1:]
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta = variables.variable('beta',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=trainable,
                                restore=restore)
    if scale:
      gamma = variables.variable('gamma',
                                 params_shape,
                                 initializer=tf.ones_initializer(),
                                 trainable=trainable,
                                 restore=restore)
    # Create moving_mean and moving_variance add them to
    # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
    moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
    moving_mean = variables.variable('moving_mean',
                                     params_shape,
                                     initializer=tf.zeros_initializer(),
                                     trainable=False,
                                     restore=restore,
                                     collections=moving_collections)
    moving_variance = variables.variable('moving_variance',
                                         params_shape,
                                         initializer=tf.ones_initializer(),
                                         trainable=False,
                                         restore=restore,
                                         collections=moving_collections)

    r_max = variables.variable('r_max',
                              (1,),
                              initializer=tf.ones_initializer(),
                              trainable=False,
                              restore=restore)
    d_max = variables.variable('d_max',
                              (1,),
                              initializer=tf.zeros_initializer(),
                              trainable=False,
                              restore=restore)
    curr_t = variables.variable('curr_t',
                                (1,),
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                restore=restore)

    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)

      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

      r_max_val = 3.0
      new_r = tf.divide(r_max_val, 1.0+(r_max_val-1.0)*tf.exp(-curr_t))
      update_r = r_max.assign(new_r)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_r)

      d_max_val = 5.0
      new_d = tf.divide(d_max_val, (1.0+(d_max_val/1e-3)-1.0)*tf.exp(-2.0*curr_t))
      update_d = d_max.assign(new_d)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_d)

      new_t = curr_t+1e-5
      update_t = curr_t.assign(new_t)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_t)

      # batch renorm
      std = tf.sqrt(variance+epsilon)
      moving_std = tf.sqrt(moving_variance+epsilon)
      r = tf.divide(std, moving_std)
      r = tf.stop_gradient(tf.clip_by_value(r, 1.0/r_max, r_max))
      
      d = tf.divide(mean - moving_mean, moving_std)
      d = tf.stop_gradient(tf.clip_by_value(d, -d_max, d_max))
      
      outputs = tf.nn.batch_normalization(
          inputs, mean, variance, None, None, epsilon) 
      outputs = tf.multiply(outputs, r) + d
        
      if scale:
        outputs = tf.multiply(outputs, gamma)
      if center:
        outputs += beta

    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance

      # Normalize the activations.
      outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    
    outputs.set_shape(inputs.get_shape())
    if activation:
      outputs = activation(outputs)
    return outputs


def _two_element_tuple(int_or_tuple):
  """Converts `int_or_tuple` to height, width.

  Several of the functions that follow accept arguments as either
  a tuple of 2 integers or a single integer.  A single integer
  indicates that the 2 values of the tuple are the same.

  This functions normalizes the input value by always returning a tuple.

  Args:
    int_or_tuple: A list of 2 ints, a single int or a tf.TensorShape.

  Returns:
    A tuple with 2 values.

  Raises:
    ValueError: If `int_or_tuple` it not well formed.
  """
  if isinstance(int_or_tuple, (list, tuple)):
    if len(int_or_tuple) != 2:
      raise ValueError('Must be a list with 2 elements: %s' % int_or_tuple)
    return int(int_or_tuple[0]), int(int_or_tuple[1])
  if isinstance(int_or_tuple, int):
    return int(int_or_tuple), int(int_or_tuple)
  if isinstance(int_or_tuple, tf.TensorShape):
    if len(int_or_tuple) == 2:
      return int_or_tuple[0], int_or_tuple[1]
  raise ValueError('Must be an int, a list with 2 elements or a TensorShape of '
                   'length 2')


@scopes.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
  """Adds a 2D convolution followed by an optional batch_norm layer.

  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input. If `batch_norm_params` is None, a
  second variable called 'biases' is added to the result of the convolution
  operation.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.

  """
  with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    conv = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                        padding=padding)
    if batch_norm_params is not None:
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [num_filters_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      outputs = activation(outputs)
    return outputs

@scopes.add_arg_scope
def depthwise_conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='VALID',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           is_norm=False,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
  """Adds a 2D depth wise convolution followed by an optional batch_norm layer.
  this applies channels differnt filters to each channel independently

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.

  """
  with tf.variable_scope(scope, 'ConvDepthWise', [inputs], reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1].value
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)

    batch_size = inputs.get_shape()[0].value
    num_pt = inputs.get_shape()[1].value

    conv = tf.nn.depthwise_conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                                 padding=padding)

    if is_norm:
        outputs = tf.reshape(conv, (batch_size*num_pt, num_filters_out, num_filters_in))
        outputs = batch_norm(outputs, decay=0.999)
        outputs = tf.reshape(conv, (batch_size, num_pt, num_filters_out, num_filters_in))
    else:
        bias_shape = [conv.get_shape()[-1],]
        bias_initializer = tf.constant_initializer(bias)
        biases = variables.variable('biases',
                                    shape=bias_shape,
                                    initializer=bias_initializer,
                                    trainable=trainable,
                                    restore=restore)
        outputs = tf.nn.bias_add(conv, biases)
        outputs = tf.reshape(outputs, (batch_size,num_pt,num_filters_out,num_filters_in))

    if activation:
      outputs = activation(outputs)
    return outputs

@scopes.add_arg_scope
def depthwise_conv2d_v1(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='VALID',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
  """Adds a 2D depth wise convolution followed by an optional batch_norm layer.
  this applies channels differnt filters to each channel independently

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
  Returns:
    a tensor representing the output of the operation.

  """
  with tf.variable_scope(scope, 'ConvDepthWise', [inputs], reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_h, kernel_w,
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    conv = tf.nn.depthwise_conv2d(inputs, weights, [1, stride_h, stride_w, 1],
                                 padding=padding)
    if batch_norm_params is not None:
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [conv.get_shape()[-1],]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      outputs = activation(outputs)
    return outputs

def _deconv_output_length(input_length, filter_size, padding, stride):
    """determines output length of a transposed convolution given input length, stride and kernel
    Args:
        padding: 'SAME', 'VALID' or 'FULL'
    Returns:
        output length
    """
    padding = padding.upper
    if input_length is None:
        return None
    input_length *= stride
    if padding == 'VALID':
        input_length += max(filter_size-stride, 0)
    elif padding == 'FULL':
        input_length -= (stride + filter_size - 2)
    return input_length

@scopes.add_arg_scope
def deconv(inputs,
           num_filters_out,
           kernel_size,
           stride,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None,
           reuse=None):
    """Adds a 2D deconvolution operator followed by optional batch_norm layer
    args:
        inputs: with size [batch_size, height, widht, channels]
        num_filters_out: number of output feature channels
        padding: 'VALID', 'SAME' 'FULL'
    returns:
        a tensor representing the output of the operation
    """
    with tf.variable_scope(scope, 'Deconv', [inputs], reuse=reuse):
        batch_size = inputs.get_shape()[0]
        height, width = inputs.get_shape()[1], inputs.get_shape()[2]
        num_filters_in = inputs.get_shape()[-1]

        kernel_h, kernel_w = _two_element_tuple(kernel_size)
        stride_h, stride_w = _two_element_tuple(stride)

        weights_shape = [kernel_h, kernel_w,
                        num_filters_out, num_filters_in]
        weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
        l2_regularizer = None
        if weight_decay and weight_decay>0:
            l2_regularizer = losses.l2_regularizer(weight_decay)
        weights = variables.variable('weights',
                                    shape=weights_shape,
                                    initializer=weights_initializer,
                                    regularizer=l2_regularizer,
                                    trainable=trainable,
                                    restore=restore)

        out_height = _deconv_output_length(height, kernel_h, padding, stride_h)
        out_width = _deconv_output_length(width, kernel_w, padding, stride_w)
        output_shape = tf.stack([batch_size, out_height, out_width, num_filters_out]) 
        deconv = tf.nn.conv2d_transpose(inputs, weights, output_shape, [1, stride_h, stride_w, 1], padding=padding)

        if batch_norm_params is not None:
            with scopes.arg_scope([batch_norm], is_training=is_training,
                                 trainable=trainable, restore=restore):
                outputs = batch_norm(deconv, **batch_norm_params)
        else:
            bias_shape = [num_filters_out,]
            bias_initializer = tf.constant_initializer(bias)
            biases = variables.variable('biases',
                                       shape=bias_shape,
                                       initializer=bias_initializer,
                                       trainable=trainable,
                                       restore=restore)
            outputs = tf.nn.bias_add(deconv, biases)
        if activation:
            outputs = activation(outputs)
            return outputs


@scopes.add_arg_scope
def fc(inputs,
       num_units_out,
       activation=tf.nn.relu,
       stddev=0.01,
       bias=0.0,
       weight_decay=0,
       batch_norm_params=None,
       is_training=True,
       trainable=True,
       restore=True,
       scope=None,
       reuse=None):
  """Adds a fully connected layer followed by an optional batch_norm layer.

  FC creates a variable called 'weights', representing the fully connected
  weight matrix, that is multiplied by the input. If `batch_norm` is None, a
  second variable called 'biases' is added to the result of the initial
  vector-matrix multiplication.

  Args:
    inputs: a [B x N] tensor where B is the batch size and N is the number of
            input units in the layer.
    num_units_out: the number of output units in the layer.
    activation: activation function.
    stddev: the standard deviation for the weights.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_scope.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.

  Returns:
     the tensor variable representing the result of the series of operations.
  """
  with tf.variable_scope(scope, 'FC', [inputs], reuse=reuse):
    num_units_in = inputs.get_shape()[1]
    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = None
    if weight_decay and weight_decay > 0:
      l2_regularizer = losses.l2_regularizer(weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    if batch_norm_params is not None:
      outputs = tf.matmul(inputs, weights)
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(outputs, **batch_norm_params)
    else:
      bias_shape = [num_units_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    if activation:
      outputs = activation(outputs)
    return outputs


def one_hot_encoding(labels, num_classes, scope=None):
  """Transform numeric labels into onehot_labels.

  Args:
    labels: [batch_size] target labels.
    num_classes: total number of classes.
    scope: Optional scope for name_scope.
  Returns:
    one hot encoding of the labels.
  """
  with tf.name_scope(scope, 'OneHotEncoding', [labels]):
    batch_size = labels.get_shape()[0]
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
    concated = tf.concat(axis=1, values=[indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, num_classes]), 1.0, 0.0)
    onehot_labels.set_shape([batch_size, num_classes])
    return onehot_labels


@scopes.add_arg_scope
def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Max Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: a list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  Raises:
    ValueError: if 'kernel_size' is not a 2-D list
  """
  with tf.name_scope(scope, 'MaxPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)

@scopes.add_arg_scope
def upsampling_nearest(inputs, scale):
    assert scale>1, 'scale of upsampling should be larger then 1'
    new_h = int(inputs.shape[1]*scale)
    new_w = int(inputs.shape[2]*scale)
    return tf.image.resize_images(inputs, [new_h, new_w], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


@scopes.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Avg Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: a list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  """
  with tf.name_scope(scope, 'AvgPool', [inputs]):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    return tf.nn.avg_pool(inputs,
                          ksize=[1, kernel_h, kernel_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding=padding)


@scopes.add_arg_scope
def dropout(inputs, keep_prob=0.5, is_training=True, scope=None):
  """Returns a dropout layer applied to the input.

  Args:
    inputs: the tensor to pass to the Dropout layer.
    keep_prob: the probability of keeping each input unit.
    is_training: whether or not the model is in training mode. If so, dropout is
    applied and values scaled. Otherwise, inputs is returned.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the output of the operation.
  """
  if is_training and keep_prob > 0:
    with tf.name_scope(scope, 'Dropout', [inputs]):
      return tf.nn.dropout(inputs, keep_prob)
  else:
    return inputs


def flatten(inputs, scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: a tensor of size [batch_size, ...].
    scope: Optional scope for name_scope.

  Returns:
    a flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: if inputs.shape is wrong.
  """
  if len(inputs.get_shape()) < 2:
    raise ValueError('Inputs must be have a least 2 dimensions')
  dims = inputs.get_shape()[1:]
  k = dims.num_elements()
  with tf.name_scope(scope, 'Flatten', [inputs]):
    return tf.reshape(inputs, [-1, k])


def repeat_op(repetitions, inputs, op, *args, **kwargs):
  """Build a sequential Tower starting from inputs by using an op repeatedly.

  It creates new scopes for each operation by increasing the counter.
  Example: given repeat_op(3, _, ops.conv2d, 64, [3, 3], scope='conv1')
    it will repeat the given op under the following variable_scopes:
      conv1/Conv
      conv1/Conv_1
      conv1/Conv_2

  Args:
    repetitions: number or repetitions.
    inputs: a tensor of size [batch_size, height, width, channels].
    op: an operation.
    *args: args for the op.
    **kwargs: kwargs for the op.

  Returns:
    a tensor result of applying the operation op, num times.
  Raises:
    ValueError: if the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  with tf.variable_scope(scope, 'RepeatOp', [inputs]):
    tower = inputs
    for _ in range(repetitions):
      tower = op(tower, *args, **kwargs)
    return tower

