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

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1,1))
        self.bn2a = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
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

        self.conv2a = tf.keras.layers.Conv2DTranspose(filters1, (1,1))
        self.bn2a = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2b = tf.keras.layers.Conv2DTranspose(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization(axis=-1)

        self.conv2c = tf.keras.layers.Conv2DTranspose(filters3, (1, 1))
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
def myModel():
    #encoder
    #1*50*50 -> 64*48*48
    inputs = tf.keras.layers.Input(shape=(50,50,1,), name="input") 
    conv0 = tf.keras.layers.Conv2D(64,9,strides=(1,1), padding="valid", activation='relu',input_shape=(50,50,1))(inputs)
    #64*48*48 ->128*48*48 ->128*24*24
    conv1 = ResnetIdentityBlock(3,[64,128,128])(conv0)
    #conv1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(conv1)
    #128*24*24 ->256*24*24->256*12*12
    conv2 = ResnetIdentityBlock(3,[128,256,256])(conv1)
    #conv2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(conv2)
    #256*12*12->512*12*12->512*6*6
    ###conv3 = ResnetIdentityBlock(3,[256,512,512])(conv2)
    #conv3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(conv3)
    #512*6*6->1024*6*6->512*6*6
    ###conv4 = tf.keras.layers.Conv2D(1024,3,strides=(1,1), padding="same", activation='relu')(conv3)
    #conv3_ = tf.keras.layers.Conv2DTranspose(512, 3, strides=(1,1), padding='same', dilation_rate=(1,1), activation='relu')(conv4)
    #conv3_ = tf.keras.layers.Add()([conv3, conv3_])
    #512*6*6->256*6*6->256*12*12
    #conv2_ = InverseResnetIdentityBlock(3,[512,256,256])(conv3_)
    #conv2_ = tf.keras.layers.UpSampling2D(size=(2,2))(conv2_)
    #conv2_ = tf.keras.layers.Add()([conv2, conv2_])
    #256*12*12->128*12*12->128*24*24
    conv1_ = InverseResnetIdentityBlock(3,[256,128,128])(conv2)
    #conv1_ = tf.keras.layers.UpSampling2D(size=(2,2))(conv1_)
    #conv1_ = tf.keras.layers.Add()([conv1, conv1_])

    conv0_ = InverseResnetIdentityBlock(3,[128,64,64])(conv1_)
    #conv0_ = tf.keras.layers.UpSampling2D(size=(2,2))(conv0_)
    #conv0_ = tf.keras.layers.Add()([conv0, conv0_])

    inputs_ = tf.keras.layers.Conv2DTranspose(1,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='sigmoid')(conv0_)
    #inputs_ = tf.keras.layers.UpSampling2D(size=(2,2))(inputs_)

    e_output = tf.keras.layers.Conv2D(1,1,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu', name='enhancementOutput')(inputs_)
    #conv5 = ResnetIdentityBlock(3,[256,512,512])(conv4_)
    
    model = tf.keras.models.Model(inputs=inputs, outputs = [e_output])
    return model
'''
def myModel():
    #encoder
    inputs = tf.keras.layers.Input(shape=(50,50,1,), name="input")
    #layer1
    conv1 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding="valid", activation='relu', input_shape=(50,50,1,))(inputs)
    #layer2
    conv2 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding="valid", activation='relu', )(conv1)
    #layer3
    conv3 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding="valid", activation='relu', )(conv2)
    i_seq = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(conv3)
    #layer4
    d_inputs = tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding="same", activation='relu',name="encoderOutput" )(i_seq)


    #decoder enhancement branch
    #e_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(d_inputs)
    e_seq = tf.keras.layers.UpSampling2D(size=(2,2))(d_inputs)
    e_seq = tf.keras.layers.Add()([e_seq, conv3])
    e_seq = tf.keras.layers.Conv2DTranspose(128,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(e_seq)
    e_seq = tf.keras.layers.Add()([e_seq, conv2])
    e_seq = tf.keras.layers.Conv2DTranspose(128,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(e_seq)
    e_seq = tf.keras.layers.Add()([e_seq, conv1])
    e_output = tf.keras.layers.Conv2DTranspose(1,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='sigmoid', name='enhancementOutput')(e_seq)


    #O_seq = tf.keras.layers.Flatten()(O_seq)
    #outputs = tf.keras.layers.Dense(2, activation='softmax')(O_seq)
    model = tf.keras.models.Model(inputs=inputs, outputs = [e_output])
    return model
'''
if __name__ == "__main__":
    model = myModel()
    model.summary()
    for layer in model.layers:
        #print("ayaya")
        if not layer.trainable:
            print(layer)
    
    dot_img_file = __file__.split('.')[0] + '.jpg'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)