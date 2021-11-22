import argparse

import cv2
import numpy as np
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from model import *

from data.ucm_dataset import UCMDataSet
from data.deepglobe import DeepGlobeDataSet

from utils.crf import DenseCRF
from utils.metric import scores

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'ucm'

MODEL = 'deeplabv2'
DATA_DIRECTORY = '../../Satellite_Images/'
DATA_LIST_PATH = '../../Satellite_Images/ImageSets/test.txt' # subset.txt
IGNORE_LABEL = 255
NUM_CLASSES = 18 # 60 for pascal context
RESTORE_FROM = './checkpoints/ucm/checkpoint_final.pth'
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'
EXP_ID = "default"
EXP_OUTPUT_DIR = './s4gan_files'

MLMT_FILE = './mlmt_output/output_ema_p_1_0_voc_5.txt'

######### CRF ################
CRF_ITER_MAX = 10 
CRF_POS_XY_STD = 1
CRF_POS_W = 3
CRF_BI_XY_STD = 1
CRF_BI_RGB_STD = 67
CRF_BI_W = 4

SAMPLING_TYPE = 'uncertainty'

MLMT_EXP_OUTPUT_DIR = './mlmt_files'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--active-learning",type=bool,default=False,
                        help="whether to use active learning to select labeled examples")
    parser.add_argument("--sampling-type", type=str, default=SAMPLING_TYPE,
                        help="sampling technique to use")
    parser.add_argument("--start-eval", type=int, default=100,
                        help="epoch from which we want to start evaluation")
    parser.add_argument("--end-eval", type=int, default=20000,
                        help="epoch upto which we want to evaluate")
    parser.add_argument("--step-eval", type=int, default=100,
                        help="evaluation steps") 
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="train,val,test,subset")
    parser.add_argument("--exp-id", type=str, default=EXP_ID,
                        help= "unique id to identify all files of an experiment")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="dataset name pascal_voc or pascal_context")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    parser.add_argument("--crf", action="store_true",
                        help="apply crf postprocessing to precomputed logits")
    parser.add_argument("--labeled-ratio", type=float, default=None,
                        help="labeled ratio of the trained model")
    parser.add_argument("--threshold-st", type=float, default=None,
                        help="threshold st of the trained model")
 
    
    return parser.parse_args()

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def UCMColorize(label_mask, save_file):
        label_colours = ucm_color_map()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 18):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = b #r
        rgb[:, :, 1] = g #g 
        rgb[:, :, 2] = r #b 
        print(save_file)
        cv2.imwrite(save_file, rgb)

        return rgb

def ucm_color_map():
        return np.asarray([[0, 0, 0], [166, 202, 240], [128, 128, 0], [0, 0, 128],
                           [255, 0, 0], [0, 128, 0], [128, 0, 0], [255, 233, 233],
                           [160, 160, 164], [0, 128, 128], [90, 87, 255], [255, 255, 0],
                           [255, 192, 0], [0, 0, 255], [255, 0, 192], [128, 0, 128],
                           [0, 255, 0], [0, 255, 255]])


def DeepGlobeColorize(label_mask, save_file):
        label_colours = deepglobe_color_map()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 7):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
        print(save_file)
        cv2.imwrite(save_file, rgb)

        return rgb

def deepglobe_color_map():
        return np.asarray([[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0],
                           [0, 0, 255], [255, 255, 255], [0, 0, 0]])

def crf_process(image, label, logit, size, postprocessor):
        H = size[0]
        W = size[1]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].cpu().numpy()
        image = torch.squeeze(image)
        image = image.numpy()
        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob_post = postprocessor(image, prob)
        crf_label = np.argmax(prob_post, axis=0)
        return crf_label

def main():
    """Create the model and start the evaluation process."""
    
    args = get_arguments()
    gpu0 = args.gpu
    start = args.start_eval
    end = args.end_eval
    step = args.step_eval

    max_mean_iou = 0.0
    max_crf_mean_iou = 0.0
    
    for epoch in range(start,end+1,step):        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        model = DeepLabV2_ResNet101_MSC(n_classes=args.num_classes)
        model.cuda()

        saved_state_dict = torch.load(os.path.join(args.checkpoint_dir))
        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu0)

        if args.dataset == 'deepglobe':
            dataset =DeepGlobeDataSet(args.data_dir, args.data_list, module='s4gan', crop_size = input_size,
                                     scale = args.random_scale, mirror = args.random_mirror, mean = IMG_MEAN)
            testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=True)

        elif args.dataset == 'ucm':
          dataset = UCMDataSet(args.data_dir, args.data_list, module = 's4gan', crop_size = input_size,
                                     scale = args.random_scale, mirror = args.random_mirror, mean = IMG_MEAN)
          testloader = data.DataLoader(dataset, batch_size = 1, shuffle = False, pin_memory = True)
          interp = nn.Upsample(size=(256,256), mode='bilinear', align_corners=True) #320, 240 # align_corners = True
            
       
        data_list = []
        gt_list = []
        output_list = []
        crf_result_list = []

     
        for index, batch in enumerate(testloader):
            if index % 1 == 0:
                print('%d processd'%(index))
            image, label, size, name, _ = batch
            size = size[0]
            output  = model(Variable(image, volatile=True).cuda(gpu0))
            crf_output = output.clone().detach()
            output = interp(output).cpu().data[0].numpy()
                
   
            
            output = output[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            
            
            if args.crf:
                postprocessor = DenseCRF(iter_max=CRF_ITER_MAX,
                                        pos_xy_std=CRF_POS_XY_STD,
                                        pos_w=CRF_POS_W,
                                        bi_xy_std=CRF_BI_XY_STD,
                                        bi_rgb_std=CRF_BI_RGB_STD,
                                        bi_w=CRF_BI_W,)
                result_crf = crf_process(image, label, crf_output, size, postprocessor)

            if args.save_output_images:
              filename = '{}.png'.format(name[0])
              savefile = os.path.join(args.viz_dir, filename)
  
              
            data_list.append([gt.flatten(), output.flatten()])

            gt_list.append(gt)
            output_list.append(output)
            if args.crf:
                crf_result_list.append(result_crf)

        score = scores(gt_list, output_list, args.num_classes)

        ####################auto-evaluate logic to evaluate scores and store the best one#########################################################
      
        mean_iou = score['Mean IoU']
        if mean_iou > max_mean_iou:
            max_mean_iou = mean_iou
            best_score = score
            best_iteration = epoch
            best_score_filename = os.path.join(scores_dir, "best_scores_" + str(epoch) + ".json")
            print('best so far...'+ str(mean_iou))
            print("Hey yo!!, Best score found at epoch : " + str(epoch) + "and the score is: " + str(max_mean_iou))
            print("best Scores saved at: ", best_score_filename)

        if args.crf:
            score_crf = scores(gt_list, crf_result_list, args.num_classes)
            crf_mean_iou = score_crf['Mean IoU']
            if crf_mean_iou > max_crf_mean_iou:
                max_crf_mean_iou = crf_mean_iou
                best_score_crf = score_crf
                best_crf_iteration = epoch
                best_score_filename_crf = os.path.join(scores_dir,"best_scores_crf_" + str(epoch) + ".json")
                print("CRF Scores saved at: ",best_score_filename_crf)

        with open(best_score_filename, "w") as f:
            json.dump(best_score, f, indent=4, sort_keys=True)


    if args.crf:
        with open(best_score_filename_crf, "w") as f:
            json.dump(best_score_crf, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
