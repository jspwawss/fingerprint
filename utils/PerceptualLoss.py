import tensorflow as tf
import numpy as np


def preprocess_image(image_path, img_ncols=50, img_nrows=50):
   
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    img_nrows = x.get_shape().as_list()[-1]
    img_ncols = x.get_shape().as_list()[-2]
    # Util function to convert a tensor into a valid image
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# The gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    x = tf.transpose(x, (0,3, 1, 2))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# The "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    img_nrows = style.get_shape().as_list()[-1]
    img_ncols = style.get_shape().as_list()[-2]
    size = img_nrows * img_ncols
    #print(type(size))
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


# An auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# The 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    '''
    img_nrows = x.get_shape().as_list()[-1]
    img_ncols = x.get_shape().as_list()[-2]
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    '''
    return tf.reduce_sum(tf.image.total_variation(x))


def perceptualLoss(feature_extractor,ori_gt, style_gt, y_pred, total_variation_weight = 1e-6,style_weight = 1e-6,content_weight = 2.5e-8):
    
    style_layer_names = ['block1_conv2','block2_conv2','block3_conv2','block4_conv2','block5_conv2']
    content_layer_name = 'block5_conv2'
    #feature_extractor.trainable = False
    batch_size = ori_gt.get_shape()[0]
    input_tensor = tf.concat([ori_gt, style_gt, y_pred], axis=0)
    features = feature_extractor(input_tensor, training=False)

    loss = tf.zeros(shape=())
    layer_features = features[content_layer_name]
    ori_features = layer_features[0:batch_size,:,:,:]
    content_features = layer_features[2*batch_size:, :,:, :]
    loss  = loss + content_weight * content_loss(ori_features, content_features)

    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[batch_size:2*batch_size,:,:,:]
        combination_features = layer_features[2*batch_size:,:,:,:]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight/len(style_layer_names))* sl
    
    loss += total_variation_weight * total_variation_loss(y_pred)

    return loss