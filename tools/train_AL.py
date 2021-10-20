import argparse
import numpy as np
import json
import pdb
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
import torchvision.models as models
from torch.utils import data

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint

import modAL
from modAL.models import ActiveLearner
from scipy.special import softmax

from data.ucm import UCMDataSet
from data.deepglobe import DeepGlobeDataSet

torch.manual_seed(360);
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
INPUT_SIZE = '320, 320'

def create_model_dict(input_size):

    model_dict = dict()
    vgg16 = models.vgg16(pretrained=True, progress=True)
    res50 = models.resnet50(pretrained=True, progress=True)
    res101 = models.resnet101(pretrained=True, progress=True)
    model_dict['vgg16'] = vgg16
    model_dict['res50'] = res50
    model_dict['res101'] = res101
    return model_dict

def create_dataset_dict(input_size):
    dataset_dict = dict()
    if args.dataset_name=='UCM': 
        dataset_dict['UCM'] = UCMDataSet(args.train_data_dir, args.train_data_list, crop_size=input_size,
                                                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                                    module='AL')
    else:
        dataset_dict['deepglobe'] = DeepGlobeDataSet(args.train_data_dir, args.train_data_list, crop_size=input_size,
                                                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN,
                                                    module='AL')
    return dataset_dict

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

#### Argument Parser
def get_arguments():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset-name", type=str, default="UCM", help="UCM/deepglobe") 
    parser.add_argument("--query-strategy", type=str, default="margin",
                                        help="uncertainty, margin, entropy sampling")
    parser.add_argument("--learning-rate", type=float, default=0.00001,
                                            help="Learning Rate")
    parser.add_argument("--batch-size", type=int, default=4,
                                            help="Batch Size")
    parser.add_argument("--num-epochs", type=int, default=10,
                                            help="Number of Epochs")
    parser.add_argument("--momentum", type=float, default=0.9,
                                                help="Momentum")
    parser.add_argument("--device", type=str, default="cuda",
                                            help="cuda/cpu")
    parser.add_argument("--random-scale", action="store_true",
                                            help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-mirror", action="store_true",
                                            help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                                            help="Comma-separated string with height and width of images.")
    parser.add_argument("--train-data-dir", type=str, default="./train",
                                            help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-data-list", type=str, default="./train/image_list.txt",
                                            help="Path to the file listing the images in the dataset.")
    parser.add_argument("--labeled-ratio", type=float, default=0.05,
                                            help="labeled ratio")
    parser.add_argument("--alpha", type=float, default=0.1,
                                            help="alpha for initial pool")
    parser.add_argument("--beta", type=float, default=0.5,
                                            help="beta for number of images to learn on")
    parser.add_argument("--N_total", type=int, default=1679,
                    help="Total Number of samples(UCM: 1679, DeepGlobe: 642")
    parser.add_argument("--model-name", type=str, default="res50", help="vgg16/res50/res101")
    parser.add_argument("--num-classes", type=int, default=21, help="UCM: 21, deepglobe: 6")
    return parser.parse_args()


args = get_arguments()

ALPHA = float(args.alpha)
BETA = float(args.beta)
dataset_name = args.dataset_name
model_name = args.model_name
N_total = args.N_total
num_classes = args.num_classes


if args.alpha<0 or args.alpha>1:
    raise ValueError('alpha should be between 0 and 1')
if args.beta<0 or args.beta>1:
    raise ValueError('beta should be between 0 and 1')

if dataset_name=='UCM': 
    if num_classes!=21:
        raise ValueError('number of classes should be equal to 21 when dataset=UCM')   
    if N_total!=1679:
        raise ValueError('total number of samples for UCM should be 1679')
    names_array = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chapparal', 'denseresidential','forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks','tenniscourt']
elif dataset_name=="deepglobe":
    if num_classes!=6:
        raise ValueError('number of classes should be equal to 6 when dataset=DeepGlobe')
    if N_total!=642:
        raise ValueError('otal number of samples for DeepGlobe should be 642')
    names_array = ['urban_land','agriculture_land','rangeland','forest','water','barren']
else:
   raise ValueError('Currently this code only supports UCM and deepglobe')

if model_name not in ['vgg16','res50','res101']:
    raise NotImplementedError('model should be either vgg16, res50, or res101')   


h, w = map(int, args.input_size.split(','))
input_size = (h, w)

#create dictionary for model and dataset
model_dict = create_model_dict(input_size)
dataset_dict = create_dataset_dict(input_size)

#Dataloader Object
train_dataset = dataset_dict[dataset_name]

names=[]

class ImageClassifier(nn.Module):
    def __init__(self):
       super(ImageClassifier,self).__init__()
       self.net = model_dict[model_name]
       self.final_layer = nn.Linear(1000,num_classes)
       self.log_softmax=nn.LogSoftmax()   

    def forward(self,x):
       x1 = self.net(x)
       y = self.final_layer(x1)
       return y


model = ImageClassifier()

#Model Callbacks
lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)
checkpoint = Checkpoint(dirname = 'exp', f_params='best_model.pt', monitor='train_loss_best')

#Define Neural Net Classifier
net = NeuralNetClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss,
    lr=args.learning_rate,
    batch_size=args.batch_size,
    max_epochs=args.num_epochs,
    optimizer=optim.SGD,
    optimizer__momentum=args.momentum,
    train_split=None,
    device=args.device # comment to train on cpu
)

#Dataloader Object
dataset = dataset_dict[dataset_name]
active_dataloader = data.DataLoader(dataset,
                        batch_size=N_total, shuffle=True, num_workers=0, pin_memory=True)#1679
X_data, y_data, _, name, _ = next(iter(active_dataloader))
name=np.asarray(name)

#Assemble initial data
'''
Assemble initial data:
1. Sample N_inital(=inital pool size) samples randomly from thee entire unlabeled pool(X_pool)
2. Remove the inital samples from X_pool
3. Query labels for these N_inital samples from the oracle
4. Train the Image Classifier
As per Algorithm:1 in the paper,
total_labeled_samples_to_query(X_L)  = labeled_ratio*total_num_samples
inital_pool_size = alpha * X_L
'''
np.random.seed(1234)
initial_pool_size = ALPHA * args.labeled_ratio * N_total
print("Initial pool size = ", initial_pool_size)
initial_idx = np.random.choice(range(len(X_train)), size=initial_pool_size, replace=False)
selected_names = list(name[initial_idx])

X_initial = X_data[initial_idx]
y_initial = y_data[initial_idx]
names_initial = name[initial_idx]

'''
Generate the pool
Remove the initial data from the training dataset
'''
X_pool = np.delete(X_data, initial_idx, axis=0)
names_pool = np.delete(name, initial_idx, axis=0)
y_pool = np.delete(y_data, initial_idx, axis=0)

#Active Learner

#Query strategy 1
if args.query_strategy == "margin":
    learner = ActiveLearner(estimator=net,
                           query_strategy=modAL.uncertainty.margin_sampling,
                           X_training=X_initial, y_training=y_initial,
                           )
#Query strategy 2
elif args.query_strategy == "entropy":
    learner = ActiveLearner(estimator=net,
                            query_strategy=modAL.uncertainty.entropy_sampling,
                            X_training=X_initial, y_training=y_initial,
                            )
else:
    raise NotImplementedError("AL sampling strategy should be either margin or query")

prediction_probabilities = []

'''
target(total_labeled_samples) = labeled_ratio * total_num_samples (In this paper, we expierment with labeled_ratio of: 0.02, 0.05, 0.125)
N_inital = alpha*target

for each query, we are querying beta*N_inital
query_samples_per_iter = np.ceil(beta*n_inital)

n_queries  = target/query_samples_per_iter
'''
print("n_initial = ", initial_pool_size)
target = math.ceil(float(args.labeled_ratio) * N_total)
print("target = ", target)
query_samples_per_iter = int(np.ceil(BETA*initial_pool_size))
print("query_samples_per_iter = ", query_samples_per_iter)
n_queries  = int(np.ceil(target/query_samples_per_iter))
print("Number of Queries: ", n_queries)


for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = learner.query(X_pool, n_instances=query_samples_per_iter)
    selected_names = list(names_pool[query_idx])
    names.extend(selected_names)
    X = X_pool[query_idx]
    prediction_prob = softmax(net.predict_proba(X_pool[query_idx]), axis=1) #0 # query_idx
    y_pred = net.predict(X_pool[query_idx]) #query_idx
    pred_class = np.argmax(prediction_prob, axis=1)
    class_prob = np.max(prediction_prob, axis=1)
    class_prob = list(class_prob)

    prediction_probabilities.extend(class_prob)
    learner.teach(
        X=X_pool[query_idx], y=y_pool[query_idx], only_new=False,
        )


    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    names_pool = np.delete(names_pool, query_idx, axis=0)


#save the name list and the prediction list:

dir_name = args.query_strategy
makedirs(dir_name)

names_arr = np.array(names[:target])
prediction_prob_arr = np.array(prediction_probabilities[:target])

names_file = os.path.join(dir_name, args.query_strategy + '_names_'+ args.labeled_ratio + '_'+ args.alpha + '_' + args.beta + '.npy')
probs_file = os.path.join(dir_name, args.query_strategy + '_probs_'+ args.labeled_ratio + '_'+ args.alpha + '_' + args.beta + '.npy')

np.save(names_file, names_arr) 
np.save(probs_file, prediction_prob_arr)    
