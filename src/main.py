import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from opts import parser
args = parser.parse_args()
dataset = __import__(args.datasetfilename, fromlist=[args.dataset])
dataset = getattr(dataset, args.dataset)
#from dataset import ljyBlur
model = __import__("arch."+args.model, fromlist=[args.model_name], level=0)
model = getattr(model, args.model_name)
model = model()
import json
from utils.LapLoss import lapLoss
from utils.crossEntropy import crossEntropy
#tf.enable_eager_execution()
print("debug=",args.debug)
#exit()
def main():
    global train_dir, val_dir
    global save_path, save_name, dir_name
    global epochs
    epochs = args.epochs if not args.debug else 10
    save_name = args.model
    #save_name = r'fingerNet_ljyBlur_v2.0'
    dir_name = list()
    dir_name.extend([args.dataset, args.model, args.save_annotation])
    if args.debug:
        dir_name.append('debug')
        os.environ['CUDA_VISIBLE_DEVICES']="-1"
    print(dir_name)
    for d_name in dir_name:
        if len(d_name) == 0:
            dir_name.remove(d_name)
    save_path = os.path.join(args.save_path, '_'.join(dir_name))
    #save_path = os.path.join(r"/home/share/Han/novatek",save_name)
    print("{0:-^50}".format(save_path))
    #exit()
    try:
        os.makedirs(save_path)
    except:
        pass
    
    if len(args.train_path)>0:
        train_dir = args.train_path

    if len(args.val_path)>0:
        val_dir = args.val_path


    '''
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        validation_split=0.0,
        rescale=1./255,
    )
    train_dataset = tf.keras.preprocessing.image.DirectoryIterator(
        train_dir,
        datagen,
        target_size=(61,61),
        color_mode="rgb",
        batch_size=1,
        shuffle=True,
        #subset="training",
        #class_mode='binary',
        class_mode='categorical',
    )
       
    train_dataset = generate_arrays_from_file(train_dir)
    '''
    if len(args.dataset_path) > 0:
        train_dataset = dataset(mode="train" ,debug = args.debug , batch_size=args.batch_size, dirpath = args.dataset_path )
        val_dataset = dataset(mode='val', debug=args.debug,  batch_size=args.batch_size, dirpath = args.dataset_path)
    else:
        train_dataset = dataset(mode="train" ,debug = args.debug , batch_size=args.batch_size, )
        val_dataset = dataset(mode='val', debug=args.debug,  batch_size=args.batch_size, )
    #print(train_dataset.__next__())
    #exit()
    train(model=model, dataset=train_dataset,val_dataset= val_dataset, epochs = epochs)
    #evaluate()
    pass
def train_epoch(model, dataset):

    pass
def generate_arrays_from_file(path:str, batch_size=2):
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


def train(model=None, dataset=None, val_dataset=None, epochs=int(10)):
    print(epochs)
    print(save_path)
    #exit()
    #loss_object  = {"orientationOutput": tf.keras.losses.categorical_crossentropy,"enhancementOutput": lapLoss}
    loss_object  = {"enhancementOutput": lapLoss}
    print("*"*50)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, )
    #model.compile(loss = loss_object, loss_weights=[1.0,1.0], optimizer=optimizer,)
    model.compile(loss = loss_object, optimizer=optimizer,)
    #model.compile(optimizer=optimizer,)
    #print("complie finish")
    #model.fit(x=dataset, epochs=epochs, batch_size=1)
    #model.fit(x=dataset['input_1'],y=[dataset['y0'], dataset['y1']], epochs=epochs, steps_per_epoch=2,)
    print(model.summary())
    print(os.path.join(save_path, save_name + r"_{epoch}-{loss:.2f}.ckpt"))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, save_name + r"_{epoch}-{loss:.2f}.h5"), save_best_only=True, monitor="val_loss",save_weights_only=True,),
            tf.keras.callbacks.CSVLogger(os.path.join(save_path, "history.log")),]

    history = model.fit(dataset, epochs=epochs, callbacks=callbacks,validation_data=val_dataset )
    print(history.history)
    print(type(history.history))
    plt.plot(history.history["loss"], label='training loss', color='blue')
    plt.plot(history.history['val_loss'], label='validate loss', color='red')
    plt.legend()
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(os.path.join(save_path, save_name + "_result.jpg"))

    del model

    
    pass
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
