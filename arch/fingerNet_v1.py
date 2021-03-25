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

def myModel():
    #encoder
    inputs = tf.keras.layers.Input(shape=(61,61,1,), name="input")
    #layer1
    i_seq = tf.keras.layers.Conv2D(64, 9, strides=(4,4), padding="valid", activation='relu', input_shape=(64,64,1,))(inputs)
    #layer2
    i_seq = tf.keras.layers.Conv2D(64, 5, strides=(1,1), padding="same", activation='relu', )(i_seq)
    #layer3
    i_seq = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding="same", activation='relu', )(i_seq)
    i_seq = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(i_seq)
    #layer4
    d_inputs = tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding="same", activation='relu',name="encoderOutput" )(i_seq)

    #decoder orientation branch
    #o_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(d_inputs)
    o_seq = tf.keras.layers.UpSampling2D(size=(2,2))(d_inputs)
    o_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="same",dilation_rate=(1,1), activation='relu')(o_seq)
    o_seq = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="same",dilation_rate=(1,1), activation='relu')(o_seq)
    o_seq = tf.keras.layers.Conv2DTranspose(64,9,strides=(4,4), padding="valid",dilation_rate=(1,1), activation='relu')(o_seq)
    o_seq = tf.keras.layers.Flatten()(o_seq)
    o_output = tf.keras.layers.Dense(21, activation="softmax", name="orientationOutput")(o_seq)

    #o_output = tf.math.argmax(o_seq)

    #decoder enhancement branch
    #e_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(d_inputs)
    e_seq = tf.keras.layers.UpSampling2D(size=(2,2))(d_inputs)
    e_seq = tf.keras.layers.Conv2DTranspose(64,3,strides=(1,1), padding="same",dilation_rate=(1,1), activation='relu')(e_seq)
    e_seq = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="same",dilation_rate=(1,1), activation='relu')(e_seq)
    e_output = tf.keras.layers.Conv2DTranspose(1,9,strides=(4,4), padding="valid",dilation_rate=(1,1), activation='sigmoid', name='enhancementOutput')(e_seq)


    #O_seq = tf.keras.layers.Flatten()(O_seq)
    #outputs = tf.keras.layers.Dense(2, activation='softmax')(O_seq)
    model = tf.keras.models.Model(inputs=inputs, outputs = [o_output, e_output])
    return model
if __name__ == "__main__":
    model = myModel()
    model.summary()
    dot_img_file = __file__.split('.')[0] + '.jpg'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)