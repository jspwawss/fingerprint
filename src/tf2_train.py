import tensorflow as tf
try:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
except:
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)

import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from opts import parser
args = parser.parse_args()
save_name = args.model
dir_name = list()
dir_name.extend([args.dataset, args.model, args.save_annotation])
if args.debug:
    dir_name.append('debug')
    print_rate = 1
    os.environ['CUDA_VISIBLE_DEVICES']="-1" #for tf1 
    CUDA_VISIBLE_DEVICES = -1 #for tf2
else:
    print_rate = args.print_rate
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    CUDA_VISIBLE_DEVICES = 0
print(dir_name)
for d_name in dir_name:
    if len(d_name) == 0:
        dir_name.remove(d_name)
save_path = os.path.join(args.save_path, '_'.join(dir_name))

dataset = __import__(args.datasetfilename, fromlist=[args.dataset])
dataset = getattr(dataset, args.dataset)
#from dataset import ljyBlur
model = __import__("arch."+args.model, fromlist=[args.model_name], level=0)
model = getattr(model, args.model_name)
model = model()
import json
from utils.LapLoss import lapLoss
from utils.crossEntropy import crossEntropy
from utils.PerceptualLoss import perceptualLoss
print("debug=",args.debug)
print('losses=',args.losses)
print('save_path=', save_path)
from utils.total_variation_regularization import total_variation_regularization, gram_matrix
#exit()

#tf.config.run_functions_eagerly(True)

def main():
    global train_dir, val_dir
    global save_path, save_name, dir_name
    global epochs
    global print_rate
    global model
    epochs = args.epochs if not args.debug else 50
    
    #save_path = os.path.join(r"/home/share/Han/novatek",save_name)
    print("{0:-^50}".format(save_path))
    if args.dataset_path:
        print('dataset = ',args.dataset_path)
    #exit()
    try:
        os.makedirs(save_path)
    except:
        pass
    
    if len(args.train_path)>0:
        train_dir = args.train_path

    if len(args.val_path)>0:
        val_dir = args.val_path


    if len(args.dataset_path) > 0:
        train_dataset = dataset(mode="train" ,debug = args.debug , batch_size=args.batch_size, dirpath = args.dataset_path )
        val_dataset = dataset(mode='val', debug=args.debug,  batch_size=args.batch_size, dirpath = args.dataset_path)
    else:
        train_dataset = dataset(mode="train" ,debug = args.debug , batch_size=args.batch_size, )
        val_dataset = dataset(mode='val', debug=args.debug,  batch_size=args.batch_size, )
    #print(train_dataset.__next__())
    #exit()
    #g = tf.Graph()
    #with g.as_default():
    vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(50,50,3), pooling=None, classes=1000,)
    vgg.trainable = False
    layer_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    
    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=layer_output_dict)
    feature_extractor.trainable = False
    #model = model()
    optimizer = tf.keras.optimizers.Adam(lr=args.lr, beta_1=args.beta_1, beta_2=args.beta_2, )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta_1, beta2=args.beta_2, epsilon=1e-08, use_locking=False,name='Adam')
    #g = tf.Graph()
    #with g.as_default():
    train(model=model,feature_extractor=feature_extractor,optimizer=optimizer, dataset=train_dataset,val_dataset= val_dataset, epochs = epochs)
    #evaluate()
    pass

def generate_arrays_from_file(path, batch_size=2):
    while 1:
        inputList = list()
        outputList = list()
        while len(inputList) < batch_size:
            for dirpath, dirnames, filenames in os.walk(path):
                print(dirpath)
                for filename in filenames:
                    print(filename)
                    if "jpg" or "png"  in filename:

                    
                        image = tf.keras.preprocessing.image.load_img(os.path.join(dirpath, filename), target_size=(61,61))
                        img = tf.keras.preprocessing.image.img_to_array(image).reshape(61,61,3)
                        
                        outputimage = tf.keras.preprocessing.image.load_img(os.path.join(dirpath, filename), target_size=(61,61))
                        output = tf.keras.preprocessing.image.img_to_array(outputimage).reshape(61,61,3)
                        label = np.array([1,0,0])
                        #yield ({"input_1": img}, {"output_1": label, "output_2": output})
                        inputList.append(img)
                        outputList.append([(label, output)])
                        if len(inputList) >= batch_size:
                            yield inputList, outputList
                            inputList = list()
                            outputList = list()

def train(model=None,feature_extractor=None, dataset=None, val_dataset=None, epochs=int(10),optimizer=None):
    print(epochs)
    print(save_path)
 
    with open(os.path.join(save_path, 'history.txt'), 'a') as txt:
        txt.write('epoch,train_losses,val_losses\n')
    train_losses, val_losses = list(), list()
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" %(epoch,))
        #model = train_epoch(model=model,dataset=dataset, optimizer=optimizer, vgg=vgg )
        train_loss, val_loss = 0, 0
        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        
            feed_dict = {'inputs':tf.constant(x_batch_train["input"]), 'style':tf.constant(y_batch_train['enhancementOutput']['style_gt']), 'content':tf.constant(y_batch_train['enhancementOutput']['ori_gt']) }
            #train_loss_ = train_epoch(model, optimizer,feature_extractor, feed_dict)
            
            
            train_loss_ = train_epoch(model=model, optimizer=optimizer, inputs=feed_dict['inputs'], style=feed_dict['style'], content=feed_dict['content'])
            #logits = model(x_batch_train, training=True)
            #loss_value = loss_fn(y_batch_train, logits)
            #print('training loss=', train_loss_)
            
            if step % print_rate == 0:
                print("[epoch {}/{}] step: {}/{}, loss: {}".format(epoch, epochs, step, len(dataset), train_loss_))
                #print("[epoch {}/{}] step: {}/{}, loss: {}|{}|{}|{}|{}".format(epoch, epochs, step, len(dataset), train_loss_, train_loss_f, train_loss_s, train_loss_tv, mse))
            train_loss += np.array(train_loss_).mean()
        train_losses.append(train_loss/len(dataset))
        for step, (x_batch_train, y_batch_train) in enumerate(val_dataset):
            feed_dict = {'inputs':x_batch_train["input"], 'style':y_batch_train['enhancementOutput']['style_gt'], 'content':y_batch_train['enhancementOutput']['ori_gt'] }
            val_loss_ = val_epoch(model, feed_dict)
            #logits = model(x_batch_train, training=True)
            #loss_value = loss_fn(y_batch_train, logits)
            #if step % 10 == 0:
            #    print("[epoch {}/{}] step: {}/{}, loss: {}".format(epoch, epochs, step, len(val_dataset), np.array(val_loss_).mean()))
            val_loss += np.array(val_loss_).mean()
        val_losses.append(val_loss/len(val_dataset))
        model.save_weights(os.path.join(save_path, save_name + r"_{epoch}-{loss:.2f}.ckpt").format(epoch=epoch, loss=val_losses[-1]))
        #saver.save(sess, os.path.join(save_path, save_name + r"_{epoch}-{loss:.2f}.pb").format(epoch=epoch, loss=val_losses[-1]))
        print("[epoch {}/{}]  loss: {}".format(epoch, epochs, val_losses[-1]))
        with open(os.path.join(save_path, 'history.txt'), 'a') as txt:
            txt.write('{epoch},{loss},{val_loss}\n'.format(epoch=epoch, loss=train_losses[-1], val_loss=val_losses[-1]))





    plt.plot(train_losses, label='training loss', color='blue')
    plt.plot(val_losses, label='validate loss', color='red')
    plt.legend()
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(os.path.join(save_path, save_name + "_result.jpg"))

    del model
@tf.function
def train_epoch(model=None, optimizer=None,feature_extractor=None, feed_dict=None, style=None, content=None, inputs=None):
    #style = feed_dict['style']
    #content= feed_dict['content']
    #inputs = feed_dict['inputs']
    with tf.GradientTape() as tape:
        #print(feed_dict)
        #tf.print(content)
        #tf.print(inputs)
        _losses = tf.zeros(shape=())
    
        outputs = model(inputs, training=True)

        #tf.print(outputs)
        
        if 'perceptual' in args.losses:
            rgb_style = tf.image.grayscale_to_rgb(style)
            rgb_outputs = tf.image.grayscale_to_rgb(outputs)
            rgb_content = tf.image.grayscale_to_rgb(content)
            ploss = perceptualLoss(feature_extractor,rgb_content,rgb_style,rgb_outputs)
            #tf.print('{0:*^50}'.format('ploss'))
            #tf.print(ploss)
            _losses += ploss
        if 'mse' in args.losses:
            mse = lapLoss(content, outputs )
            mse = tf.keras.losses.MeanSquaredError()(content, outputs)
            #tf.print('{0:*^50}'.format('mse'))
            #tf.print(mse)
            _losses += mse
        if 'enMSE' in args.losses:
            enMSE = lapLoss(style, outputs)
            #tf.print('{0:*^50}'.format('enMSE'))
            #tf.print(enMSE)
            _losses += enMSE

        #train_vars = model.trainable_variables
        if 'l1' in args.losses:
            l1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)(content, outputs)
            _losses += l1
            #print('{0:*^50}'.format('l1'))
            #print(l1)
        
    gradients = tape.gradient(_losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #print(_losses)
    return _losses

@tf.function
def val_epoch(model,feed_dict):
    style = feed_dict['style']
    content= feed_dict['content']
    inputs = feed_dict['inputs']

    pred = model(inputs, training=False)
    _losses = tf.zeros(shape=())
    if 'perceptual' in args.losses:
        rgb_style = tf.image.grayscale_to_rgb(style)
        rgb_outputs = tf.image.grayscale_to_rgb(pred)
        rgb_content = tf.image.grayscale_to_rgb(content)
        ploss = perceptualLoss(rgb_content,rgb_style,rgb_outputs)
        _losses += ploss
    if 'mse' in args.losses:
        mse = lapLoss(content, pred )
        _losses += mse
    if 'enMSE' in args.losses:
        enMSE = lapLoss(style, pred)
        _losses += enMSE

    train_vars = model.trainable_variables
    if 'l1' in args.losses:
        l1 = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)(content, pred)
        _losses += l1
    return _losses

def evaluate():
    model = tf.keras.models.load_model(os.path.join(save_path, "fingerNet_v0_2-0.02"))
    model.summary()
    input = np.random.randint(0,255, size=(1,61,61,1))
    print(input.shape)
    output = model.predict(input)
    print(output)
    print(output[1].shape)
    pass
def evaluate_epoch():
    pass

if __name__=="__main__":
    main()
