import cv2
import numpy as np
import json
import random
import os.path as osp
from torch.utils import data



class DeepGlobeDataSet(data.Dataset):
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
        
        self.class_map = json.load(open('class_map.json','r'))
        self.class_mappings = {'urban_land':0,'agriculture_land':1,'rangeland':2,'forest_land':3,'water':4,'barren_land':5}    
        

        for name in self.img_ids:
            img_file = osp.join(self.root, "DeepGlobe_Images/%s_sat.jpg" % name)
            
            if self.module == 's4gan':
                label_file = osp.join(self.root, "DeepGlobe_Labels/%s_mask.png" % name)
            else:
                class_id = self.class_map[name]
                label_file = self.class_mappings[class_id]

            
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


    def get_deepglobe_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (5, 3)
        """

        return np.asarray(
            [
               [0, 255, 255],#urban_land
               [255, 255, 0], #agriculture_land
               [255, 0, 255], #rangeland
               [0, 255, 0], #forest
               [0, 0, 255] #water 
            ]
        )     

     #https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py#L140
    def encode_segmap(self, mask):
        """Encode segmentation label images as deepglobe classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
               (M, N, 3), in which the Pascal classes are encoded as colours.
            Returns:
             (np.ndarray): class map with dimensions (M,N), where the value at
             a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_deepglobe_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
          
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], -1)
        image = cv2.resize(image, (320,320), interpolation=cv2.INTER_CUBIC)
     
        if self.module == 's4gan':
            label = cv2.imread(datafiles["label"])
            label = cv2.resize(label, (320,320), interpolation=cv2.INTER_CUBIC)
            label = self.encode_segmap(label)
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
                                         value = (0.0, 0.0, 0.0))
    
            if self.module == 's4gan':
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value = (self.ignore_label,))
            else:
                label_pad = label
        else:
            img_pad, label_pad = image, label

        img_h, img_w = img_pad.shape[0], img_pad.shape[1]
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        if self.module == 's4gan':
            label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            if self.module == 's4gan':
                label = label[:, ::flip]
        return image.copy(), label.copy(), np.array(size), name, index
