'''
2017 scienceDirect
cite from Deep convolutional neural network for latent fingerprint enhancement
Jian Li et al.
doi: 10.1016/j.image.2017.08.010

'''

import tensorflow as tf

class Model(tf.keras.Model):
    @staticmethod
    def encoder(inputs):
        #encoder

        #layer1
        O_seq = tf.keras.layers.Conv2D(64, 9, strides=(4,4), padding="valid", activation='relu', input_shape=(64,64,3))(inputs)
        #layer2
        O_seq = tf.keras.layers.Conv2D(64, 5, strides=(1,1), padding="valid", activation='relu', )(O_seq)
        #layer3
        O_seq = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding="valid", activation='relu', )(O_seq)
        O_seq = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(O_seq)
        #layer4
        d_inputs = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding="valid", activation='relu', )(O_seq)
        return d_inputs

    @staticmethod
    def orientation_branch(d_inputs, num_orientation=20):
        #decoder orientation branch
        o_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(d_inputs)
        o_seq = tf.keras.layers.UpSampling2D(size=(2,2))(o_seq)
        o_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(o_seq)
        o_seq = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(o_seq)
        o_seq = tf.keras.layers.Conv2DTranspose(64,9,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(o_seq)
        o_output = tf.keras.layers.Dense(9, activation="softmax")(o_seq)
        return o_output

    @staticmethod
    def enhancement_branch(d_inputs):
        #decoder enhancement branch
        e_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(d_inputs)
        e_seq = tf.keras.layers.UpSampling2D(size=(2,2))(o_seq)
        e_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(e_seq)
        e_seq = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(e_seq)
        e_output = tf.keras.layers.Conv2DTranspose(64,9,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(e_seq)
        return e_output
    @staticmethod
    def build(width=61, height=61, num_orientation=20):
        input_shape = (width, height, 3)
        inputs = tf.keras.layers.Input(shape=input_shape)
        d_inputs = Model.encoder(inputs)
        orientation_output = Model.orientation_branch(d_inputs, num_orientation=num_orientation)
        enhancement_output = Model.enhancement_branch(d_inputs)

        model = tf.keras.models.Model(inputs=inputs, outputs = [orientation_output, enhancement_output], name="fingerNet")
        return model
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (3,3), padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2c = tf.keras.layers.Conv2D(filters3, (3,3), padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization(axis=-1)

        self.relu = tf.keras.layers.ReLU()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return self.relu(x)
class InverseResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(InverseResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2DTranspose(filters1, (3,3), padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2b = tf.keras.layers.Conv2DTranspose(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2c = tf.keras.layers.Conv2DTranspose(filters3, (3,3), padding='same')
        self.bn2c = tf.keras.layers.BatchNormalization(axis=-1)

        self.relu = tf.keras.layers.ReLU()
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = self.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return self.relu(x)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var

def myModel():
    #encoder
    #1*50*50 -> 64*48*48
    inputs = tf.keras.layers.Input(shape=(50,50,1,), name="input") 
    edges = tf.image.sobel_edges(inputs)
    edge_x = clip_0_1(edges[...,0])
    edge_y = clip_0_1(edges[...,1])
    edge = tf.keras.layers.Add()([edge_x, edge_y])
    conv0 = tf.keras.layers.Conv2D(64,9,strides=(1,1), padding="valid", activation='relu')(edge)

    #conv1 = tf.keras.layers.Conv2D(64,5,strides=(1,1), padding="valid", activation='relu')(conv0)
    conv1 = ResnetIdentityBlock(9,[64,64,64])(conv0)
    conv2 = ResnetIdentityBlock(7,[64,64,64])(conv1)
    conv3 = ResnetIdentityBlock(5,[64,64,64])(conv2)

    conv4 = ResnetIdentityBlock(3,[64,64,64])(conv3)

    conv3_ = InverseResnetIdentityBlock(3,[64,64,64])(conv4)
    #conv2_ = tf.keras.layers.Add()([conv2,conv2_])
    conv2_ = InverseResnetIdentityBlock(5,[64,64,64])(conv3_)
    #conv1_ = tf.keras.layers.Add()([conv1,conv1_])
    #conv0_ = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(conv1_)
    conv1_ = InverseResnetIdentityBlock(7,[64,64,64])(conv2_)
    conv0_ = InverseResnetIdentityBlock(9,[64,64,64])(conv1_)
    #conv0_ = tf.keras.layers.Add()([conv0,conv0_])
    e_output = tf.keras.layers.Conv2DTranspose(1,9,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(conv0_)
    inputs_ = tf.keras.layers.Add()([edge,e_output])
    inputs_ = tf.keras.layers.Conv2D(64,9 , padding='same')(inputs_)

    #inputs_ = tf.keras.layers.Conv2DTranspose(1,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(conv0_0)

    output = tf.keras.layers.Conv2D(1,3,strides=(1,1), padding="same",dilation_rate=(1,1), activation='relu', name='enhancementOutput')(inputs_)
    #conv5 = ResnetIdentityBlock(3,[256,512,512])(conv4_)
    
    model = tf.keras.models.Model(inputs=inputs, outputs = [output])
    return model

if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES']="-1"

    model = myModel()
    model.summary()
    variables_names = [v.name for v in tf.trainable_variables()]
    with tf.Session() as sess:
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            #print(v)
        #exit()
    for layer in model.layers:
        #print("ayaya")
        if not layer.trainable:
            print(layer)
        
    dot_img_file = __file__.split('.')[0] + '.png'
    #print( __file__)
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)