#import tensorflow as tf
import numpy as np
import os
import cv2
import tensorflow as tf
import sys
sys.path.append(os.getcwd())
from utils import FingerprintImageEnhancer
import ast
import json

class kiaraNoise(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraNoise")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"noise_in")
        self.enDir = os.path.join(self.dirpath, r"enhanced_GT")
        self.gtDir = os.path.join(self.dirpath, r"GT")
        self.mode = 'a' if mode == "train" else 'b'
        print(self.noiseDir, self.enDir,self.gtDir  )
        self.noise_files = list()
        self.gt_files = list()
        self.en_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                #people = file.split('.')[0].split("_")[:-1]
                #people = "_".join(people) + '.tif'
                people = file
                if os.path.exists(os.path.join(self.enDir.format(part=i, mode=self.mode), people)):
                    if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                        self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
                        self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                        self.en_files.append(os.path.join(os.path.join(self.enDir.format(part=i, mode=self.mode), people)))
            self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        self.enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()
        self.debug=debug
        print('self.debug', self.debug)
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        if not self.debug:
            return len(self.data) // self.batch_size
        else:
            return 1

    def __getitem__(self, idx):
        noise_files = list()
        gt_files = list()
        en_files = list()
        for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]:
            noise_files.append(file[0])
            gt_files.append(file[1])
            en_files.append(file[2])
        noise_files = np.asarray(noise_files)
        gt_files = np.asarray(gt_files)
        en_files = np.asarray(en_files)
        #noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = list()
        ori_gt = list()
        style_gt = list()
        for noise_file, gt_file, en_file in zip(noise_files, gt_files, en_files):
            _noise_file = cv2.imread(noise_file,0)
            _noise_file = cv2.resize(_noise_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _noise_file = _noise_file.reshape(50,50,1)
            noise.append(_noise_file)

            _gt_file = cv2.imread(gt_file,0)
            _gt_file = cv2.resize(_gt_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _gt_file = _gt_file.reshape(50,50,1)
            ori_gt.append(_gt_file)

            _en_file = cv2.imread(en_file,0)
            _en_file = 1-(cv2.resize(_en_file,(50,50), interpolation=cv2.INTER_AREA)/255)
            _en_file = _en_file.reshape(50,50,1)
            style_gt.append(_en_file)
            if self.debug:
                cv2.imwrite('/home/nmsoc/FPR/Han/fingerprint/testData/noise_file.jpg', _noise_file*255)
                cv2.imwrite('/home/nmsoc/FPR/Han/fingerprint/testData/gt_file.jpg', _gt_file*255)
                cv2.imwrite('/home/nmsoc/FPR/Han/fingerprint/testData/en_file.jpg', _en_file*255)

        #noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        #gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
        noise = np.asarray(noise, dtype=np.float32)
        ori_gt = np.asarray(ori_gt,dtype=np.float32)
        style_gt = np.asarray(style_gt, dtype=np.float32)
        
        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        return ({"input": noise}, {"enhancementOutput": {"ori_gt": ori_gt, "style_gt": style_gt}})

        #gt = np.asarray([ori_gt, style_gt])
        #return ({"input": noise}, {"enhancementOutput":gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file, e_file in zip(self.noise_files, self.gt_files, self.en_files):
            self.data.append([n_file,g_file, e_file])
    def get_orientation(self, ):
        
        with open(self.orient_path, 'r') as txt:
            self.orientDict = json.load(txt)
        '''
        for line in lines:
            db = line.split(',')[0]

            name = line.split(',')[1]
            orient = float(line.split(',')[-1])
            self.orientDict.update({name:orient})
        '''

class kiaraBlurAndNoise(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraBlurAndNoise")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"blur_and_noise_in")
        self.enDir = os.path.join(self.dirpath, r"GT")
        self.gtDir = dirpath.replace('noise_patch', 'blur')
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        self.en_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                #people = file.split('.')[0].split("_")[:-1]
                #people = "_".join(people) + '.tif'
                people = file
                if os.path.exists(os.path.join(self.enDir.format(part=i, mode=self.mode), people)):
                    if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                        self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
                        self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                        self.en_files.append(os.path.join(os.path.join(self.enDir.format(part=i, mode=self.mode), people)))
            self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        self.enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()
        self.debug=debug
        print('self.debug', self.debug)
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        if not self.debug:
            return len(self.data) // self.batch_size
        else:
            return 10

    def __getitem__(self, idx):
        noise_files = list()
        gt_files = list()
        en_files = list()
        for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]:
            noise_files.append(file[0])
            gt_files.append(file[1])
            en_files.append(file[2])
        noise_files = np.asarray(noise_files)
        gt_files = np.asarray(gt_files)
        en_files = np.asarray(en_files)
        #noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = list()
        ori_gt = list()
        style_gt = list()
        for noise_file, gt_file, en_file in zip(noise_files, gt_files, en_files):
            _noise_file = cv2.imread(noise_file,0)
            _noise_file = cv2.resize(_noise_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _noise_file = _noise_file.reshape(50,50,1)
            noise.append(_noise_file)

            _gt_file = cv2.imread(gt_file,0)
            _gt_file = cv2.resize(_gt_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _gt_file = _gt_file.reshape(50,50,1)
            ori_gt.append(_gt_file)

            _en_file = cv2.imread(en_file,0)
            _en_file = cv2.resize(_en_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _en_file = _en_file.reshape(50,50,1)
            style_gt.append(_en_file)

        #noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        #gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
        noise = np.asarray(noise, dtype=np.float32)
        ori_gt = np.asarray(ori_gt,dtype=np.float32)
        style_gt = np.asarray(style_gt, dtype=np.float32)
        
        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        return ({"input": noise}, {"enhancementOutput": {"ori_gt": ori_gt, "style_gt": style_gt}})

        #gt = np.asarray([ori_gt, style_gt])
        #return ({"input": noise}, {"enhancementOutput":gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file, e_file in zip(self.noise_files, self.gt_files, self.en_files):
            self.data.append([n_file,g_file, e_file])
    def get_orientation(self, ):
        
        with open(self.orient_path, 'r') as txt:
            self.orientDict = json.load(txt)
        '''
        for line in lines:
            db = line.split(',')[0]

            name = line.split(',')[1]
            orient = float(line.split(',')[-1])
            self.orientDict.update({name:orient})
        '''
class kiaraNoise4wholeStyle(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraNosie4wholeStyle")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"noise_in")
        
        self.enDir = dirpath.replace('noise_patch', 'enhanced')
        self.gtDir = dirpath.replace('noise_patch', 'blur')
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        self.en_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                people = file.split('.')[0].split("_")[:-1]
                people = "_".join(people) + '.tif'
                #people = file
                if os.path.exists(os.path.join(self.enDir.format(part=i, mode=self.mode), people)):
                    if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                        self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
                        self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                        self.en_files.append(os.path.join(os.path.join(self.enDir.format(part=i, mode=self.mode), people)))
        self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        self.enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()
        self.debug=debug
        print('self.debug', self.debug)
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        if not self.debug:
            return len(self.data) // self.batch_size
        else:
            return 10

    def __getitem__(self, idx):
        noise_files = list()
        gt_files = list()
        en_files = list()
        for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]:
            noise_files.append(file[0])
            gt_files.append(file[1])
            en_files.append(file[2])
        noise_files = np.asarray(noise_files)
        gt_files = np.asarray(gt_files)
        en_files = np.asarray(en_files)
        #noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = list()
        ori_gt = list()
        style_gt = list()
        for noise_file, gt_file, en_file in zip(noise_files, gt_files, en_files):
            _noise_file = cv2.imread(noise_file,0)
            _noise_file = cv2.resize(_noise_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _noise_file = _noise_file.reshape(50,50,1)
            noise.append(_noise_file)

            _gt_file = cv2.imread(gt_file,0)
            _gt_file = cv2.resize(_gt_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _gt_file = _gt_file.reshape(50,50,1)
            ori_gt.append(_gt_file)

            _en_file = cv2.imread(en_file,0)
            _en_file = cv2.resize(_en_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _en_file = _en_file.reshape(50,50,1)
            style_gt.append(_en_file)

        #noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        #gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
        noise = np.asarray(noise, dtype=np.float32)
        ori_gt = np.asarray(ori_gt,dtype=np.float32)
        style_gt = np.asarray(style_gt, dtype=np.float32)
        
        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        return ({"input": noise}, {"enhancementOutput": {"ori_gt": ori_gt, "style_gt": style_gt}})

        #gt = np.asarray([ori_gt, style_gt])
        #return ({"input": noise}, {"enhancementOutput":gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file, e_file in zip(self.noise_files, self.gt_files, self.en_files):
            self.data.append([n_file,g_file, e_file])
    def get_orientation(self, ):
        
        with open(self.orient_path, 'r') as txt:
            self.orientDict = json.load(txt)
        '''
        for line in lines:
            db = line.split(',')[0]

            name = line.split(',')[1]
            orient = float(line.split(',')[-1])
            self.orientDict.update({name:orient})
        '''
class kiaraBlurAndNoise4wholeStyle(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraBlurAndNoise4wholeStyle")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"blur_and_noise_in")
        self.enDir = dirpath.replace('noise_patch', 'enhanced')
        self.gtDir = dirpath.replace('noise_patch', 'blur')
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        self.en_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                people = file.split('.')[0].split("_")[:-1]
                people = "_".join(people) + '.tif'
                #people = file
                if os.path.exists(os.path.join(self.enDir.format(part=i, mode=self.mode), people)):
                    if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                        self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
                        self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                        self.en_files.append(os.path.join(os.path.join(self.enDir.format(part=i, mode=self.mode), people)))
        self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        self.enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()
        self.debug=debug
        print('self.debug', self.debug)
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        if not self.debug:
            return len(self.data) // self.batch_size
        else:
            return 10

    def __getitem__(self, idx):
        noise_files = list()
        gt_files = list()
        en_files = list()
        for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]:
            noise_files.append(file[0])
            gt_files.append(file[1])
            en_files.append(file[2])
        noise_files = np.asarray(noise_files)
        gt_files = np.asarray(gt_files)
        en_files = np.asarray(en_files)
        #noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = list()
        ori_gt = list()
        style_gt = list()
        for noise_file, gt_file, en_file in zip(noise_files, gt_files, en_files):
            _noise_file = cv2.imread(noise_file,0)
            _noise_file = cv2.resize(_noise_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _noise_file = _noise_file.reshape(50,50,1)
            noise.append(_noise_file)

            _gt_file = cv2.imread(gt_file,0)
            _gt_file = cv2.resize(_gt_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _gt_file = _gt_file.reshape(50,50,1)          
            ori_gt.append(_gt_file)

            _en_file = cv2.imread(en_file,0)
            _en_file = cv2.resize(_en_file,(50,50), interpolation=cv2.INTER_AREA)/255
            _en_file = _en_file.reshape(50,50,1)
            style_gt.append(_en_file)

        #noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        #gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
        noise = np.asarray(noise, dtype=np.float32)
        ori_gt = np.asarray(ori_gt,dtype=np.float32)
        style_gt = np.asarray(style_gt, dtype=np.float32)
        
        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        return ({"input": noise}, {"enhancementOutput": {"ori_gt": ori_gt, "style_gt": style_gt}})

        #gt = np.asarray([ori_gt, style_gt])
        #return ({"input": noise}, {"enhancementOutput":gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file, e_file in zip(self.noise_files, self.gt_files, self.en_files):
            self.data.append([n_file,g_file, e_file])
    def get_orientation(self, ):
        
        with open(self.orient_path, 'r') as txt:
            self.orientDict = json.load(txt)
        '''
        for line in lines:
            db = line.split(',')[0]

            name = line.split(',')[1]
            orient = float(line.split(',')[-1])
            self.orientDict.update({name:orient})
        '''


if __name__ == "__main__":

    dataset = kiaraNoise(mode='train' ,dirpath='/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}')

    print("dataset py")
    print(dataset)
    print(len(dataset))

    val_dataset = kiaraNoise(mode='val',dirpath='/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}')
    print(len(val_dataset))
    #exit()
    for data in dataset:
        print(data)
        print(data[0]["input"].shape)
        print(data[1]["enhancementOutput"]['ori_gt'].shape)
        print(data[1]['enhancementOutput']['style_gt'].shape)
        break

