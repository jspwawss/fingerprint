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
class FVCdataset(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/To_hanhan/dataset_Ver_0/", batch_size=10):
        print("using FVSdataset")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"cropped_add_speckle_noise")
        self.gtDir = os.path.join(self.dirpath, r"cropped_GT")
        self.noise_files = [os.path.join(self.noiseDir, file) for file in os.listdir(self.noiseDir) if os.path.isfile(os.path.join(self.noiseDir, file))]
        self.gt_files = [os.path.join(self.gtDir, file) for file in os.listdir(self.gtDir) if os.path.isfile(os.path.join(self.gtDir, file))]
        self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        return len(self.data) // self.batch_size
        #return 10

    def __getitem__(self, idx):
        #print("__getitem__")
        #print(idx*self.batch_size, (idx+1)*self.batch_size)
        #print(self.data[idx*self.batch_size: (idx+1)*self.batch_size])
        noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #print("0000")
        noise = np.asarray([(cv2.imread(noise_file, 0).reshape(61,61,1))/255 for noise_file in noise_files])
        gt = np.asarray([(cv2.imread(gt_file, 0).reshape(61,61,1))/255 for gt_file in gt_files])
        #print("1111")
        return ({"input": noise}, {"orientationOutput": np.array([[1.]+[0.]*19]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        #return ({"input": noise}, {"orientationOutput": [[1.]+[0.]*19]*self.batch_size, "enhancementOutput": gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("pepper_noised", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "pepper_noised")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for file in self.noise_files:
            self.data.append([file, file.replace("pepper_noised", "GT").replace("cropped_add_speckle_noise", "cropped_GT")])


class ljyBlur(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/blur/Db{part}_{mode}/",mode="train", batch_size=10):
        print("using ljyBlur")
        self.dirpath = dirpath

        self.orient_path = self.dirpath.split(r'/')[:-2]
        print(self.orient_path)
        self.orient_path = os.sep.join(self.orient_path)
        self.orient_path = os.path.join(self.orient_path,'orientation_v3.json')
        self.noiseDir = os.path.join(self.dirpath, r"in")
        self.gtDir = os.path.join(self.dirpath, r"GT")
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        for i in range(1,5):
            #print(self.noiseDir.format(part=i, mode=self.mode))
            #print(self.gtDir.format(part=i, mode=self.mode))
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                if os.path.isfile(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file)):

                    self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                    self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
        #print(len(self.noise_files))
        #self.noise_files = [os.path.join(self.noiseDir, file) for file in os.listdir(self.noiseDir) if os.path.isfile(os.path.join(self.noiseDir, file))]
        #self.gt_files = [os.path.join(self.gtDir, file) for file in os.listdir(self.gtDir) if os.path.isfile(os.path.join(self.gtDir, file))]
        self.batch_size = batch_size
        if len(self.noise_files) != len(self.gt_files):
            self.check()
        self.data= list()
        self.prepareData()
        self.enhancer = FingerprintImageEnhancer.FingerprintImageEnhancer()
        print("orient path=",self.orient_path)
        if os.path.isfile(self.orient_path):
            print("orientaion exist")
            self.get_orientation()
        #print("__init__ fininsh")

    def __len__(self):
        #print("__len__")
        #return len(self.data) // self.batch_size
        return 10

    def __getitem__(self, idx):
        #print("__getitem__")
        #print(idx*self.batch_size, (idx+1)*self.batch_size)
        #print(self.data[idx*self.batch_size: (idx+1)*self.batch_size])
        noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #print("0000")
        #img = cv2.resize(cv2.imread(noise_files[0], 0),(61,61),interpolation = cv2.INTER_AREA)/255
        #print(img)
        #print(img.shape)
        noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(61,61),interpolation = cv2.INTER_AREA)/255).reshape(61,61,1) for noise_file in noise_files])
        gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(61,61),interpolation = cv2.INTER_AREA)/255).reshape(61,61,1) for gt_file in gt_files])
        orientation = list()
        if not os.path.isfile(self.orient_path):
            for _gt in gt:
                try:    #can get orientation
                    _orientation = self.enhancer.get_orientation(_gt) if self.enhancer.get_orientation(_gt) >=0 else self.enhancer.get_orientation(_gt)+np.pi
                    _orientation = _orientation/np.pi*180
                    _orientation = _orientation//9
                    _orient = np.zeros(21)
                    #print(_orient)
                    #print(_orientation)
                    ###_orient[int(_orientation)] = 1
                except:#empty image or something else
                    _orient = np.zeros(21)
                orientation.append(_orient)
        else:
            #print('read orientation from dictionary')
            for gt_file in gt_files:
                filename = gt_file.split('/')[-1]
                db = gt_file.split('/')[-3]
                _orientation = self.orientDict[db][filename]
                #print("db={},filename={}".format(db,filename))
                _orient = np.zeros(21)
                #print(_orient)
                '''
                if not isinstance(_orientation, str):


                    _orient[int(_orientation)] = 1
                else:
                    _orient[-1] = 1
                '''
                #print(_orient)
                orientation.append(_orient)
        orientation = np.asarray(orientation)

        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        return ({"input": noise}, {"orientationOutput": orientation, "enhancementOutput": gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for file in self.noise_files:
            self.data.append([file, file.replace("in", "GT")])
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

class kiaraNoise(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraNoise")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"noise_in")
        self.gtDir = os.path.join(dirpath.replace('noise_patch', 'blur'), "GT")
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                if os.path.isfile(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file)):

                    self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                    self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
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
        noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
 

        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        #return ({"input": noise}, {"orientationOutput": orientation, "enhancementOutput": gt})

      
        return ({"input": noise}, {"enhancementOutput": gt})

    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("noise_in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "noise_in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file in zip(self.noise_files, self.gt_files):
            self.data.append([n_file,g_file])
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
class kiaraNoise_v0(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraNoise_v0 (this is for fake orientation)")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"in")
        self.gtDir = os.path.join(self.dirpath, r"GT")
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                if os.path.isfile(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file)):

                    self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                    self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
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
        noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(61,61),interpolation=cv2.INTER_AREA)/255).reshape(61,61,1) for noise_file in noise_files])
        gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(61,61),interpolation = cv2.INTER_AREA)/255).reshape(61,61,1) for gt_file in gt_files])
        orientation = list()
        for gt_file in gt_files:
            
            _orient = np.zeros(21)
            #print(_orient)
            '''
            if not isinstance(_orientation, str):


                _orient[int(_orientation)] = 1
            else:
                _orient[-1] = 1
            '''
            #print(_orient)
            orientation.append(_orient)
        orientation = np.asarray(orientation)
      
        return ({"input": noise}, {"orientationOutput": orientation, "enhancementOutput": gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("noise_in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "noise_in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file in zip(self.noise_files, self.gt_files):
            self.data.append([n_file,g_file])
            
class kiaraBlurAndNoise(tf.keras.utils.Sequence):
    def __init__(self, dirpath= r"/home/share/FVC/FVC2000/noise_patch/Db{part}_{mode}/",mode="train", batch_size=10 , debug=False):
        print("using kiaraBlurAndNoise")
        self.dirpath = dirpath
        self.noiseDir = os.path.join(self.dirpath, r"in")
        self.gtDir = r'/home/share/FVC/FVC2000/blur/Db{part}_{mode}/GT/'
        self.mode = 'a' if mode == "train" else 'b'
        self.noise_files = list()
        self.gt_files = list()
        for i in range(1,5):
            for file in os.listdir(self.noiseDir.format(part=i, mode=self.mode)):
                if os.path.isfile(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file)):

                    self.noise_files.append(os.path.join(self.noiseDir.format(part=i, mode=self.mode), file))
                if os.path.isfile(os.path.join(self.gtDir.format(part=i, mode=self.mode), file)):
                    self.gt_files.append(os.path.join(self.gtDir.format(part=i, mode=self.mode), file))
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
        noise_files= np.asarray([file[0] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        gt_files = np.asarray([file[1] for file in self.data[idx*self.batch_size: (idx+1)*self.batch_size]])
        #noise = np.asarray([(cv2.imread(noise_file, 0)/255).reshape(50,50,1) for noise_file in noise_files])
        noise = np.asarray([(cv2.resize(cv2.imread(noise_file, 0),(50,50),interpolation=cv2.INTER_AREA)/255).reshape(50,50,1) for noise_file in noise_files])
        gt = np.asarray([(cv2.resize(cv2.imread(gt_file, 0),(50,50),interpolation = cv2.INTER_AREA)/255).reshape(50,50,1) for gt_file in gt_files])
 

        #print("1111")
        ##return ({"input": noise}, {"orientationOutput": np.array([[0.]*20]*self.batch_size).reshape(-1,20), "enhancementOutput": gt})
        #return ({"input": noise}, {"orientationOutput": orientation, "enhancementOutput": gt})

      
        return ({"input": noise}, {"enhancementOutput": gt})



    def check(self):
        for noise_file in self.noise_files:
            if not os.path.isfile(noise_file.replace("noise_in", "GT")):
                self.noise_files.remove(noise_file)
        for gt_file in self.gt_files:
            if not os.path.isfile(gt_file.replace("GT", "noise_in")):
                self.gt_files.remove(gt_file)
    def prepareData(self):
        for n_file, g_file in zip(self.noise_files, self.gt_files):
            self.data.append([n_file,g_file])

class kiaraNoise4perceptual(tf.keras.utils.Sequence):
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
        return 1
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
        return ({"input": noise, 'en_gt': style_gt, 'content': ori_gt}, {"enhancementOutput": ori_gt})

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
    dataset = kiaraNoise(mode='train', dirpath='/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}/')

    print("dataset py")
    print(dataset)
    print(len(dataset))

    val_dataset = kiaraNoise(mode='val', dirpath='/home/nmsoc/FPR/FVC2000/noise_patch/Db{part}_{mode}/')
    print(len(val_dataset))
    #exit()
    for data in dataset:
        print(data)
        print(data[0]["input"].shape)
        print(data[1]["orientationOutput"].shape)
        print(data[1]['enhancementOutput'].shape)
        break

