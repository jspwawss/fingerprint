import tensorflow as tf
import numpy as np
#do NOT enable, will crash while saving model
#tf.enable_eager_execution()

def crossEntropy(y_true,y_pred):
    ce = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0,
    name='categorical_crossentropy'
    )
    y_true = y_true.reshape(-1,21)
    y_pred = y_true.reshape(-1,21)
    return ce(y_true,y_pred)



if __name__ == "__main__":
    import numpy as np
    Loss = crossEntropy
    y_true = np.array([[0,1],[1,0],[1,0]], dtype=float)
    y_pred = np.array([[1,1],[0,0],[1,0]], dtype=float)

    print(np.asarray([y_true,y_pred]))
    loss = Loss(y_true,y_pred)
    print(loss)
    