  import os
  import os.path as osp
  import numpy as np
  import random
  #import matplotlib.pyplot as plt
  import collections
  import torch
  import torchvision
  import cv2
  from torch.utils import data
  from PIL import Image
  import re


class DeepGlobeDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, module):

        self.module = module
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        
        self.files = []    
        
        self.class_map = json.load(open('/home/dg777/project/Satellite_Images/DeepGlobeImageSets/class_map.json','r'))
        self.class_mappings = {'urban_land':0,'agriculture_land':1,'rangeland':2,'forest_land':3,'water':4,'barren_land':5}    
        

        for name in self.img_ids:
            img_file = osp.join(self.root, "UCMerced_Images/%s.tif" % name)
            
            if self.module == 's4gan':
                label_file = osp.join(self.root, "UCMerced_Labels/%s.png" % name)
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
         np.ndarray with dimensions (7, 3)
      """

       return np.asarray(
           [
               [0, 255, 255],#urban_land
               [255, 255, 0], #agriculture_land
               [255, 0, 255], #rangeland
               [0, 255, 0], #forest
               [0, 0, 255], #water 
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
     image = cv2.resize(image, (256,256), interpolation=cv2.INTER_CUBIC)
     
     if self.module == 's4gan':
         label = cv2.imread(datafiles["label"])
         label = cv2.resize(label, (320,320), interpolation=cv2.INTER_CUBIC)
         label = self.encode_segmap(label)
     else:
         label = np.asarray(datafiles["label"])

     size = image.shape
     name = datafiles["name"]
    
     if self.scale:
         image, label = self.generate_scale_label(image, la
     image = np.asarray(image, np.float32)
     image -= self.mean
     image = image.transpose((2, 0, 1))

     if self.is_mirror:
         flip = np.random.choice(2) * 2 - 1
         image = image[:, :, ::flip]
     return image.copy(), label.copy(), np.array(size), name, index
