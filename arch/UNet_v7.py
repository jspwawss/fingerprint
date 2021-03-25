import tensorflow as tf

class PerceptualModel(tf.keras.Model):
    def train_step(self, data):
        print("in train step")
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            #loss = self.perceptualLoss(y,y_pred) + self.compiled_loss(y_pred, y[0])
            loss = self.perceptualLoss(y,y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    def perceptualLoss(self, y, y_pred):
        ori_gt = y['enhancementOutput']['ori_gt']
        style_gt = y['enhancementOutput']['style_gt']
        
        rgb_ori_gt = tf.image.grayscale_to_rgb(ori_gt)
        rgb_style_gt = tf.image.grayscale_to_rgb(style_gt)
        rgb_prd = tf.image.grayscale_to_rgb(y_pred)
        vgg = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(50,50,3), pooling=None, classes=1000,)
        vgg.trainable = False
        '''
        style_layer1_output = vgg.get_layer('block1_conv2').output
        style_layer2_output = vgg.get_layer('block2_conv2').output
        style_layer3_output = vgg.get_layer('block3_conv2').output

        content_output = vgg.get_layer('block3_conv2').output

        style_model = tf.keras.Model(vgg.input, outputs=[style_layer1_output,style_layer2_output,style_layer3_output])
        content_model = tf.keras.Model(vgg.input, outputs=[content_output])

        style_gt_tensor1, style_gt_tensor2, style_gt_tensor3 = style_model.predict(rgb_style_gt)
        style_pd_tensor1, style_pd_tensor2, style_pd_tensor3 = style_model.predict(rgb_prd)

        content_gt_tensor = content_model.predict(rgb_ori_gt)
        content_pd_tensor = style_pd_tensor3
        style_gt_tensor = np.array([style_gt_tensor1, style_gt_tensor2, style_gt_tensor3])
        style_pd_tensor = np.array([style_pd_tensor1, style_pd_tensor2, style_pd_tensor3])

        style_loss = tf.keras.losses.mean_squared_error(style_gt_tensor,style_pd_tensor)
        content_loss = tf.keras.losses.mean_squared_error(content_gt_tensor, content_pd_tensor)
        print('style_loss',style_loss)
        print("content_loss",content_loss)
        #return style_loss+content_loss+tf.keras.losses.mean_squared_error(ori_gt_gray, y_pred)
        #return style_loss+content_loss
        '''
        return tf.keras.losses.mean_squared_error(ori_gt_gray, y_pred)



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
def myModel(input_shape = (50,50,3)):
    #encoder
    #1*50*50 -> 64*48*48
    inputs = tf.keras.layers.Input(shape=(50,50,1,), name="input") 
    conv0 = tf.keras.layers.Conv2D(64,9,strides=(1,1), padding="valid", activation='relu')(inputs)

    #conv1 = tf.keras.layers.Conv2D(64,5,strides=(1,1), padding="valid", activation='relu')(conv0)
    conv1 = ResnetIdentityBlock(9,[64,64,64])(conv0)
    conv2 = ResnetIdentityBlock(5,[64,64,64])(conv1)
    conv3 = ResnetIdentityBlock(3,[64,64,64])(conv2)

    conv4 = ResnetIdentityBlock(3,[64,64,64])(conv3)

    conv3_ = InverseResnetIdentityBlock(3,[64,64,64])(conv4)
    #conv2_ = tf.keras.layers.Add()([conv2,conv2_])
    conv2_ = InverseResnetIdentityBlock(3,[64,64,64])(conv3_)
    #conv1_ = tf.keras.layers.Add()([conv1,conv1_])
    #conv0_ = tf.keras.layers.Conv2DTranspose(64,5,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(conv1_)
    conv1_ = InverseResnetIdentityBlock(5,[64,64,64])(conv2_)
    conv0_ = InverseResnetIdentityBlock(9,[64,64,64])(conv1_)
    #conv0_ = tf.keras.layers.Add()([conv0,conv0_])
    e_output = tf.keras.layers.Conv2DTranspose(1,9,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu', name='enhancementOutput')(conv0_)
    #inputs_ = tf.keras.layers.Add()([inputs,inputs_])

    #inputs_ = tf.keras.layers.Conv2DTranspose(1,3,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu')(conv0_0)

    #e_output = tf.keras.layers.Conv2D(1,1,strides=(1,1), padding="valid",dilation_rate=(1,1), activation='relu', name='enhancementOutput')(inputs_)
    #conv5 = ResnetIdentityBlock(3,[256,512,512])(conv4_)
    '''
    style_input = tf.keras.layers.Input(shape=input_shape, name='style_input')

    vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=style_input,
    input_shape=input_shape, pooling=None, classes=1000,)
    vgg.trainable = False
    for layers in vgg.layers:
        print(layers)
    print(vgg.summary())
    style_layer1_output = vgg.get_layer('block1_conv2').output
    style_layer2_output = vgg.get_layer('block2_conv2').output
    style_layer3_output = vgg.get_layer('block3_conv2').output

    con
    '''
    #model = tf.keras.models.Model(inputs=inputs, outputs = [e_output])
    model = PerceptualModel(inputs=inputs, outputs=[e_output])
    return model

if __name__ == "__main__":
    model = myModel()
    model.summary()
    for layer in model.layers:
        #print("ayaya")
        if not layer.trainable:
            print(layer)
    
    dot_img_file = __file__.split('.')[0] + '.jpg'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)