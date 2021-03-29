import numpy as np
import tensorflow as tf
from utils.LapLoss import lapLoss, total_loss
from utils.PerceptualLoss import perceptualLoss
from arch.UNet_v8 import myModel
import os
try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
except:
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)

os.environ['CUDA_VISIBLE_DEVICES']="0"
CUDA_VISIBLE_DEVICES = 0

model = myModel()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')
 

    def call(self, x):
        x = self.conv1(x)

        return x


#model = MyModel()
#model.build((None,50,50,1,))
print(model.summary())
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(50,50,3), pooling=None, classes=1000,)
vgg.trainable = False
layer_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=layer_output_dict)
feature_extractor.trainable = False

@tf.function
def train_step(images, labels,input_model=None, optimizer=None, feed_dict=None):
    print('train_step')
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        rgb_style = tf.image.grayscale_to_rgb(labels)
        rgb_outputs = tf.image.grayscale_to_rgb(predictions)
        rgb_content = tf.image.grayscale_to_rgb(labels)
        ploss = perceptualLoss(feature_extractor,rgb_content,rgb_style,rgb_outputs)
        #tf.print(predictions)
        #loss = loss_object(labels, predictions)
        #loss = tf.keras.losses.MeanSquaredError()(predictions, labels)
        #loss = total_loss(predictions, labels)
        loss = lapLoss(predictions, labels)
    loss += ploss
    gradients = tape.gradient(loss, model.trainable_variables)
    print(ploss)
    print(loss)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    #train_loss(loss)
    return loss
    #train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    input_shape = (10, 50,50,1)
    x = tf.random.uniform(input_shape)
    y = tf.random.uniform(input_shape)
    #print(x)
    #print(y)
    #pred = model(x)
    #print('*'*50)
    #print(pred)
    #print('*'*50)
    loss = train_step(x, y,input_model=model, optimizer=optimizer)
    print('loss=',loss)
    '''
    exit()
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
    '''