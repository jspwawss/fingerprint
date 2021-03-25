import tensorflow as tf
import numpy as np
#do NOT enable, will crash while saving model
#tf.enable_eager_execution()
'''
class LapLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.LapKernel = tf.keras.initializers.Constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], shape=[3,3,1,1])
        self.conv2d = tf.keras.layers.Conv2D(1,3,use_bias=False, kernel_initializer=self.LapKernel)
        self.alpha = 0.2
    def call(self, y_true, y_pred):
        trueLap = self.conv2d(y_true)
        preLap = self.con2d(y_pred)
        return  self.alpha * tf.keras.losses.mean_squared_error(trueLap, preLap) + (1-self.alpha)*tf.keras.losses.MeanSquaredError(y_true,y_pred)
'''
def lapLoss(y_true,y_pred):
    
    #alpha = tf.keras.initializers.Constant(0.2)
    alpha = 0.2
    #LapKernel = tf.keras.initializers.Constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], )(shape=[3,3,1,1])
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape(3,3,1,1)
    #print(kernel)
    #conv2d = tf.keras.layers.Conv2D(1,3,use_bias=False, kernel_initializer=kernel, )
    #conv2d = tf.keras.layers.Conv2D(1,3,use_bias=False, )
    #conv2d.build((None, ))
    #conv2d.set_weights(kernel)
    #trueLap = conv2d(y_true)
    #preLap = conv2d(y_pred)
    trueLap = tf.nn.conv2d(y_true, filter=kernel, strides=[1,1,1,1], padding='SAME')
    preLap = tf.nn.conv2d(y_pred, filter=kernel, strides=[1,1,1,1], padding='SAME')
    trueLap.trainable = False
    preLap.trainable=False
    #y_true = y_true['enhancementOutput']['ori_gt']
    #trueLap = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=y_true.shape[1:])(y_true)
    #preLap = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=y_pred.shape[1:])(y_pred)
    #print("{0:*^50}".format('trueLap'))
    #print(trueLap)
    #print("{0:*^50}".format('preLap'))
    #print(preLap)
    #print("{0:*^50}".format('y_true'))
    #print(y_pred)
    #print("{0:*^50}".format('y_pred'))
    #print(y_true)
    #print("*"*50)
    MeanSquaredError = tf.keras.losses.MeanSquaredError()
    # if test ==> .numpy()
    lap = MeanSquaredError(trueLap, preLap)
    mse = MeanSquaredError(y_true,y_pred)
    #print("{0:*^50}".format('lap'))
    #print(lap)
    #print("{0:*^50}".format('mse'))
    #print(mse)
    #print(type(trueLap))
    #print(type(y_true))
    
    return alpha * lap+ mse*(1-alpha)


if __name__ == "__main__":
    import numpy as np
    import cv2
    import os

    pred_place = tf.placeholder(tf.float32, shape=[None, 50,50,1])
    gt_place = tf.placeholder(tf.float32, shape=[None, 50,50,1])
    loss = lapLoss(gt_place,pred_place)
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    with tf.Session() as sess:
        pred = (cv2.imread(r'/home/nmsoc/FPR/Han/fingerprint/testData/UNet_v6_debug_percep_0.png',0)).reshape(1,50,50,1)/255
        gt = cv2.imread(r'/home/nmsoc/FPR/Han/fingerprint/testData/gt_file.jpg', 0).reshape(1,50,50,1)/255
        print(pred.shape)
        print(gt.shape)
        feed_dict = {pred_place: pred, gt_place:gt}
        _loss = sess.run(loss, feed_dict=feed_dict)
        print(_loss)
        