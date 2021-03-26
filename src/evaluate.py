import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")

import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
#from utils.LapLoss import lapLoss


parser.add_argument("--save_path", type=str, default=r"/home/share/Han/novatek")
parser.add_argument("--save_weight_only", action='store_true')
parser.add_argument("--custom_objects",action='append',)
parser.add_argument('--testing_data',type=str, default='/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}/')
parser.add_argument('--checkpoint_path', type=str, default=r'/home/share/Han/novatek/fingerNet_ljyBlur_v1.3_lap/fingerNet_ljyBlur_v1.3_97-0.06.ckpt')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--model_name', type=str, default='myModel')
parser.add_argument('--input_shape', type=int, default=50)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--using_CPU", action="store_true")
parser.add_argument('--datasetfilename', type=str, default='dataset')
parser.add_argument('--dataset', type=str, default='kiaraNoise')

args = parser.parse_args()
#tf.enable_eager_execution()

if args.using_CPU:
    print('using CPU')
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
#checkpoint_path = r'/home/share/Han/novatek/fingerNet_ljyBlur_v0_noLabel/fingerNet_v0_99-0.01'
#checkpoint_path = r'/home/share/Han/novatek/fingerNet_ljyBlur_v1.3_lap/fingerNet_ljyBlur_v1.3_97-0.06.ckpt'
if args.save_weight_only:
    model = __import__("arch."+args.model, fromlist=[args.model_name], level=0).myModel()
    model.load_weights(args.checkpoint_path)
else:
    model = tf.keras.models.load_model(checkpoint_path, custom_objects={"lapLoss":lapLoss})
model.summary()
dot_img_file = __file__.split('.')[0] + '.jpg'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
if 'ljy' in args.testing_data:
    testimgs = [r'/home/nmsoc/FPR/FVC2000/blur/Db1_b/GT/101_1_2.png',	r'/home/nmsoc/FPR/FVC2000/blur/Db1_b/in/101_1_2.png',
    r'/home/nmsoc/FPR/FVC2000/blur/Db2_b/GT/101_1_10.png',	r'/home/nmsoc/FPR/FVC2000/blur/Db2_b/in/101_1_10.png',
    r'/home/nmsoc/FPR/FVC2000/blur/Db3_b/GT/101_1_28.png',	r'/home/nmsoc/FPR/FVC2000/blur/Db3_b/in/101_1_28.png',
    r'/home/nmsoc/FPR/FVC2000/blur/Db4_b/GT/101_1_0.png',	r'/home/nmsoc/FPR/FVC2000/blur/Db4_b/in/101_1_0.png',
    ]
elif 'kiaraBlurAndNoise' in args.testing_data:
    testimgs = [r'/home/nmsoc/FPR/FVC2000/noise_patch/Db1_b/GT/101_1_2.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db1_b/blur_and_noise_in/101_1_2.png',
    r'/home/nmsoc/FPR/FVC2000/noise_patch/Db2_b/GT/101_1_10.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db2_b/blur_and_noise_in/101_1_10.png',
    r'/home/nmsoc/FPR/FVC2000/noise_patch/Db3_b/GT/101_1_28.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db3_b/blur_and_noise_in/101_1_28.png',
    r'/home/nmsoc/FPR/FVC2000/blur/Db4_b/GT/101_1_0.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db4_b/blur_and_noise_in/101_1_0.png',
    ]
elif 'kiara' in args.testing_data:
    testimgs = [r'/home/nmsoc/FPR/FVC2000/noise_patch/Db1_b/GT/101_1_2.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db1_b/noise_in/101_1_2.png',
    r'/home/nmsoc/FPR/FVC2000/noise_patch/Db2_b/GT/101_1_10.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db2_b/noise_in/101_1_10.png',
    r'/home/nmsoc/FPR/FVC2000/noise_patch/Db3_b/GT/101_1_28.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db3_b/noise_in/101_1_28.png',
    r'/home/nmsoc/FPR/FVC2000/blur/Db4_b/GT/101_1_0.png',	r'/home/nmsoc/FPR/FVC2000/noise_patch/Db4_b/noise_in/101_1_0.png',
    ]
elif '{part}' in args.testing_data:
    dataset = __import__(args.datasetfilename)
    dataset = getattr(dataset, args.dataset)
    testimgs = dataset(mode="train" ,debug = args.debug , batch_size=1,dirpath = args.testing_data )
else:
    testimgs = args.testing_data
if not args.debug:
    if len(testimgs) == 8:
        skip_ratio = 2
        start = 1
    else:
        skip_ratio = 1
        start = 0
    for testimg in testimgs[start::skip_ratio]:
        #testimg = r'/home/share/FVC/FVC2000/blur/Db1_b/GT/101_1_2.png'
        img = (cv2.resize(cv2.imread(testimg, 0),(args.input_shape, args.input_shape),interpolation = cv2.INTER_AREA)/255).reshape(1,args.input_shape, args.input_shape,1)
        if "fingerNet" in args.model:
            output = model.predict(img)[1]
        else:
            output = model.predict(img)
        print(output.shape)
        cv2.imwrite(os.path.join(os.getcwd(),'testData','_'.join([args.model, args.testing_data,'pred', ''])+testimg.split('/')[-1]), output[0]*255)
    
else:
    for step, (x_batch_train, y_batch_train) in enumerate(testimgs):
        output = model.predict(x_batch_train['input'])
        output_tensor = tf.constant(output)
        output_rgb = tf.image.grayscale_to_rgb(output_tensor)
        print(output.shape)
        #print(y_batch_train)
        if 'perceptual' in args.datasetfilename:
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'pred',str(step)+'.png'])), output[0]*255)
            
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'pred_rgb',str(step)+'.png'])), output_rgb.eval(session=tf.compat.v1.Session())[0]*255)
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'content',str(step)+'.png'])), y_batch_train['enhancementOutput']['ori_gt'][0]*255)
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'style',str(step)+'.png'])), y_batch_train['enhancementOutput']['style_gt'][0]*255)

        else:
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'pred',str(step)+'.png'])), output[0]*255)
            cv2.imwrite(os.path.join(os.getcwd(), 'testData', '_'.join([args.model, 'debug', 'output',str(step)+'.png'])), y_batch_train['enhancementOutput'][0]*255)
        '''
path = r'/home/share/FVC/FVC2000/blur/Db{part}_{mode}/in/'
output_path = r'/home/share/FVC/FVC2000/forTesting/Han/fingerNet_ljyBlur_v0_noLabel/Db{part}_{mode}/'
for part in range(1,5):
    for mode in ['a', 'b']:
        dirPath = path.format(part=part, mode=mode)
        if not os.path.exists(output_path.format(part=part, mode=mode)):
            os.makedirs(output_path.format(part=part, mode=mode))
        for dirpath, dirnames, filenames in os.walk(dirPath):
            for filename in filenames:
                testimg = os.path.join(dirpath, filename)
                img = (cv2.resize(cv2.imread(testimg, 0),(61,61),interpolation = cv2.INTER_AREA)/255).reshape(1,61,61,1)
                output = model.predict(img)[1]
                #print(output.shape)
                cv2.imwrite(os.path.join(output_path.format(part=part, mode=mode), filename), output[0]*255)
                #exit()
'''
print('test pass ^^')