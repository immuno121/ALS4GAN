import cv2
import numpy as np
import random
import os.path as osp
from torch.utils import data
from PIL import Image
import re

class UCMDataSet(data.Dataset):
    def __init__(self, root, list_path, module, crop_size=(320, 320), mean=(128, 128, 128), scale=False, mirror=False, ignore_label=255):

        self.module = module
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        
        self.files = []    
        self.class_map = {'agricultural':0, 'airplane':1, 'baseballdiamond':2, 'beach':3, 'buildings':4, 'chaparral':5,
                        'denseresidential':6, 'forest':7, 'freeway':8, 'golfcourse':9, 'harbor':10, 'intersection':11,
                        'mediumresidential':12, 'mobilehomepark':13, 'overpass':14, 'parkinglot':15, 'river':16,
                        'runway':17,'sparseresidential':18, 'storagetanks':19, 'tenniscourt':20}
       
        for name in self.img_ids:
            img_file = osp.join(self.root, "UCMerced_Images/%s.tif" % name)
            
            if self.module == 's4gan':
                label_file = osp.join(self.root, "UCMerced_Labels/%s.png" % name)
            else:
                template = re.compile("([a-zA-Z]+)([0-9]+)") 
                class_name = template.match(name).groups()[0] 
                label_file = self.class_map[class_name]

            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
           })     
         
       
    def __len__(self):    
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        if self.module == 's4gan':
            label = cv2. resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], -1)
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
        if self.module == 's4gan':
            label = np.asarray(Image.open(datafiles["label"]), dtype=np.int32)
        else:
            label = np.asarray(datafiles["label"])

        size = image.shape
        name = datafiles["name"]
    
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        
        if self.module == 's4gan':
            img_h, img_w = label.shape
        else:
            img_h = image.shape[0]
            img_w = image.shape[1]
     
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
             pad_w, cv2.BORDER_CONSTANT,
             value=(0.0, 0.0, 0.0))
                
            if self.module == 's4gan':
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(self.ignore_label,))
            else:
                label_pad = label
        else:
            img_pad, label_pad = image, label

        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
     
        if self.module == 's4gan':
            label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            if self.module == 's4gan':
                label = label[:, ::flip]
    
        return image.copy(), label.copy(), np.array(size), name, index
