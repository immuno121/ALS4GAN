import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
import pdb
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from utils.metric import scores
#from model.deeplabv2 import Res_Deeplab
from model import *
#from model.deeplabv3p import Res_Deeplab
from data.voc_dataset import VOCDataSet
from data.ucm_dataset import UCMDataSet
from data.deepglobe import DeepGlobeDataSet
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.crf import DenseCRF

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATASET = 'ucm' # pascal_context

MODEL = 'deeplabv2' # deeeplabv2, deeplabv3p
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
    parser.add_argument("--with-mlmt", action="store_true",
                        help="combine with Multi-Label Mean Teacher branch")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    parser.add_argument("--crf", action="store_true",
                        help="apply crf postprocessing to precomputed logits")
    parser.add_argument("--labeled-ratio", type=float, default=None,
                        help="labeled ratio of the trained model")
    parser.add_argument("--threshold-st", type=float, default=None,
                        help="threshold st of the trained model")
    parser.add_argument("--mlmt-file", type = str, default = MLMT_FILE,
                        help = "Where MLMT output")
 
    
    return parser.parse_args()

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

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



def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(args, data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
 
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()


    if args.dataset == 'pascal_voc':
        classes = np.array(('background',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'))
    elif args.dataset == 'pascal_context':
        classes = np.array(('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet' , 'ceiling', 'cloth', 'computer', 'cup',
                'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate',
                'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood'))
    elif args.dataset == 'cityscapes':
        classes = np.array(("road", "sidewalk",
            "building", "wall", "fence", "pole",
            "traffic_light", "traffic_sign", "vegetation",
            "terrain", "sky", "person", "rider",
            "car", "truck", "bus",
            "train", "motorcycle", "bicycle")) 
    elif args.dataset == 'ucm':
        classes = np.array(('background',  # always index 0
            'airplane', 'bare_soil', 'buildings', 'cars',
            'chapparal', 'court', 'dock', 'field', 'grass',
            'mobile_home', 'pavement', 'sand', 'sea',
            'ship', 'tanks', 'trees',
            'water'))


    for i, iou in enumerate(j_list):
         if j_list[i] > 0:
            print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

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

        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        #elif EXP_ID == "default":
        #    print("Restoring from default")
        #    saved_state_dict = torch.load(os.path.join(RESTORE_FROM))
        elif args.threshold_st is not None:
            print("Loading new weights")
            print(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train",str(args.labeled_ratio), str(args.threshold_st) ,'checkpoint'+str(epoch)+'.pth'), 'saved weights')
            saved_state_dict = torch.load(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train",str(args.labeled_ratio), str(args.threshold_st) ,'checkpoint'+str(epoch)+'.pth'))

        else:
            print("Loading old weights")
            #saved_state_dict = torch.load(args.restore_from)
            print(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train", 'checkpoint'+str(epoch)+'.pth'), 'saved weights')
            saved_state_dict = torch.load(os.path.join(EXP_OUTPUT_DIR, "models", args.exp_id, "train", 'checkpoint'+str(epoch)+'.pth'))
            #print("Restoring from: ", saved_state_dict)
        
        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda(gpu0)

        if args.dataset == 'pascal_voc':
            testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(320, 240), mean=IMG_MEAN, scale=False, mirror=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(320, 240), mode='bilinear', align_corners=True)

        elif args.dataset == 'deepglobe':
            testloader = data.DataLoader(DeepGlobeDataSet(args.data_dir, args.data_list, args.active_learning, args.labeled_ratio, args.sampling_type,  crop_size=(320, 320), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=True)

        elif args.dataset == 'ucm':
            testloader = data.DataLoader(UCMDataSet(args.data_dir, args.data_list, args.active_learning, args.labeled_ratio, args.sampling_type, crop_size=(256,256), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(256,256), mode='bilinear', align_corners=True) #320, 240 # align_corners = True
            if args.crf:
                testloader = data.DataLoader(UCMDataSet(args.data_dir, args.data_list, crop_size=(256, 256), mean=IMG_MEAN, scale=False, mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True)
                interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True) #320, 240


        elif args.dataset == 'pascal_context':
            input_transform = transform.Compose([transform.ToTensor(),
                    transform.Normalize([.485, .456, .406], [.229, .224, .225])])
            data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': 512}
            data_loader = get_loader('pascal_context')
            data_path = get_data_path('pascal_context')
            test_dataset = data_loader(data_path, split='val', mode='val', **data_kwargs)
            testloader = data.DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1, pin_memory=True)
            interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

        elif args.dataset == 'cityscapes':
            data_loader = get_loader('cityscapes')
            data_path = get_data_path('cityscapes')
            test_dataset = data_loader( data_path, img_size=(512, 1024), is_transform=True, split='val')
            testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
            interp = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
                    
        data_list = []
        gt_list = []
        output_list = []
        crf_result_list = []


        if args.with_mlmt:
            mlmt_preds = np.loadtxt(args.mlmt_file, dtype = float)
            
            mlmt_preds[mlmt_preds>=0.2] = 1
            mlmt_preds[mlmt_preds<0.2] = 0 
             
        for index, batch in enumerate(testloader):
            if index % 1 == 0:
                print('%d processd'%(index))
            image, label, size, name, _ = batch
            size = size[0]
            output  = model(Variable(image, volatile=True).cuda(gpu0))
            crf_output = output.clone().detach()
            output = interp(output).cpu().data[0].numpy()
                
            if args.dataset == 'pascal_voc':
                output = output[:,:size[0],:size[1]]
                gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            elif args.dataset == 'ucm':
                output = output[:,:size[0],:size[1]]
                gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            elif args.dataset == 'deepglobe':
                output = output[:,:size[0],:size[1]]
                gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            elif args.dataset == 'pascal_context':
                gt = np.asarray(label[0].numpy(), dtype=np.int)
            elif args.dataset == 'cityscapes':
                gt = np.asarray(label[0].numpy(), dtype=np.int)

            if args.with_mlmt:
                for i in range(args.num_classes):
                    output[i]= output[i]*mlmt_preds[index][i]
                    
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            if args.labeled_ratio is not None:
                viz_dir = os.path.join(EXP_OUTPUT_DIR, "output_viz", args.exp_id, args.dataset_split, str(args.labeled_ratio), str(args.threshold_st))
                viz_dir_crf = os.path.join(EXP_OUTPUT_DIR, "output_viz", args.exp_id,  args.dataset_split, "crf", str(args.labeled_ratio), str(args.threshold_st))
            else:
                viz_dir = os.path.join(EXP_OUTPUT_DIR, "output_viz", args.exp_id, args.dataset_split)
                viz_dir_crf = os.path.join(EXP_OUTPUT_DIR, "output_viz", args.exp_id,  args.dataset_split, "crf")

            if not os.path.exists(viz_dir):
                makedirs(viz_dir)
            if args.crf:
                if not os.path.exists(viz_dir_crf):
                    makedirs(viz_dir_crf)
            #print("Visualization dst:", viz_dir)
            if args.crf:
                postprocessor = DenseCRF(iter_max=CRF_ITER_MAX,
                                        pos_xy_std=CRF_POS_XY_STD,
                                        pos_w=CRF_POS_W,
                                        bi_xy_std=CRF_BI_XY_STD,
                                        bi_rgb_std=CRF_BI_RGB_STD,
                                        bi_w=CRF_BI_W,)
                result_crf = crf_process(image, label, crf_output, size, postprocessor)
            if args.save_output_images:
                if args.dataset == 'pascal_voc' or  args.dataset == 'ucm':
                    filename = '{}.png'.format(name[0])
                    savefile = os.path.join(viz_dir, filename)
                    color_file = UCMColorize(output, savefile)
                if args.dataset == 'deepglobe':
                    filename = '{}.png'.format(name[0])
                    savefile = os.path.join(viz_dir, filename)
                    color_file = DeepGlobeColorize(output, savefile)


                if args.crf:
                    filename = '{}.png'.format(name[0])
                    savefile = os.path.join(viz_dir_crf, filename)
                    color_file = UCMColorize(result_crf, savefile)
                    
            data_list.append([gt.flatten(), output.flatten()])

            gt_list.append(gt)
            output_list.append(output)
            if args.crf:
                crf_result_list.append(result_crf)

        if args.labeled_ratio is not None:
            scores_dir = os.path.join(EXP_OUTPUT_DIR, "scores", args.exp_id, args.dataset_split, str(args.labeled_ratio), str(args.threshold_st))
        else:
            scores_dir = os.path.join(EXP_OUTPUT_DIR, "scores", args.exp_id, args.dataset_split)
        if not os.path.exists(scores_dir):
            makedirs(scores_dir)

        score = scores(gt_list, output_list, args.num_classes)
        print(score)
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
