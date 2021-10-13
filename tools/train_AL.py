import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import math
from torchvision import datasets, models, transforms
import torchvision.models as models
from torch.utils import data

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint

import os
import argparse

from data.ucm_dataset import UCMDataSet
import modAL
from modAL.models import ActiveLearner
from scipy.special import softmax

torch.manual_seed(360);
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
INPUT_SIZE = '320, 320'

PARENT_PATH = '/home/dg777/project/Satellite_Images'


TRAIN_DATA_DIRECTORY = '/home/dg777/project/Satellite_Images'
TRAIN_DATA_LIST_PATH = '/home/dg777/project/Satellite_Images/UCMImageSets/train.txt' # TODO: MAKE NEW TEXT FILE
TEST_DATA_DIRECTORY = '/home/dg777/project/Satellite_Images'
TEST_DATA_LIST_PATH = '/home/dg777/project/Satellite_Images/UCMImageSets/test.txt' # TODO: MAKE NEW TEXT FILE

def create_model_dict():

    model_dict = dict()
    vgg16 = models.vgg16(pretrained=True, progress=True)
    res50 = models.resnet50(pretrained=True, progress=True)
    res101 = models.resnet101(pretrained=True, progress=True)
    model_dict['vgg16'] = vgg16
    model_dict['res50'] = res50
    model_dict['res101'] = res101
    return model_dict

#### Argument Parser
def get_arguments():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--dataset", type=str, default="UCM", help="UCM/deepglobe") 
    parser.add_argument("--query-strategy", type=str, default="uncertainty",
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
    parser.add_argument("--train-data-dir", type=str, default=TRAIN_DATA_DIRECTORY,
                                            help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train-data-list", type=str, default=TRAIN_DATA_LIST_PATH,
                                            help="Path to the file listing the images in the dataset.")
    parser.add_argument("--test-data-dir", type=str, default=TEST_DATA_DIRECTORY,
                                            help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--test-data-list", type=str, default=TEST_DATA_LIST_PATH,
                                            help="Path to the file listing the images in the dataset.") 
    parser.add_argument("--labeled-ratio", type=str, default="0.05",
                                            help="labeled ratio")
    parser.add_argument("--alpha", type=str, default="0.1",
                                            help="alpha for initial pool")
    parser.add_argument("--beta", type=str, default="0.5",
                                            help="beta for number of images to learn on")
    parser.add_argument("--N_total", type=int, default="1679",
                                            help="Total Number of samples")
    parser.add_argument("--model", type=str, default="res50", help="vgg16/res50/res101")
    parser.add_argument("--num-classes", type=int, default=21, help="UCM: 21, deepglobe: 6")
    return parser.parse_args()


 args = get_arguments()

ALPHA = float(args.alpha)
BETA = float(args.beta)
dataset = args.dataset
model_name = args.model_name
N_total = args.N_total
num_classes = args.num_classes

model_dict = create_model_dict()


initial_pool_size = ALPHA * args.labeled_ratio * N_total

if dataset=='UCM': 
    names_array = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chapparal', 'denseresidential','forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks','tenniscourt']
else:
    names_array = ['urban_land','agriculture_land','rangeland','forest','water','barren']

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

h, w = map(int, args.input_size.split(','))
input_size = (h, w)

#### Dataloader Object
train_dataset = UCMDataSet(args.train_data_dir, args.train_data_list, crop_size=input_size,
                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
test_dataset = UCMDataSet(args.test_data_dir, args.test_data_list, crop_size=input_size,
                                                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

trainloader = data.DataLoader(train_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

testloader = data.DataLoader(test_dataset,
                                batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

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

#### Model Callbacks
lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)

checkpoint = Checkpoint(dirname = 'exp', f_params='best_model.pt', monitor='train_loss_best')

#### Neural Net Classifier
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
#### Train the network

active_ucm_dataloader = data.DataLoader(train_dataset,
                        batch_size=N_total, shuffle=True, num_workers=0, pin_memory=True)#1679
(X_train,name), y_train = next(iter(active_ucm_dataloader))
name=np.asarray(name)

#### Split X and y into seed and pool

dir_name = args.query_strategy
makedirs(dir_name)

# assemble initial data
np.random.seed(1234) 
n_initial = initial_pool_dict[args.labeled_ratio]
print("Initial pool size = ", n_initial)
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
selected_names = list(name[initial_idx])
print(selected_names, 'selected names')

X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
names_initial = name[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)
names_pool = np.delete(name, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

#### Active Learner

# QUERY strategy 1
# initialize ActiveLearner
if args.query_strategy == "uncertainty":
    learner = ActiveLearner(estimator=net,
                            query_strategy=modAL.uncertainty.uncertainty_sampling,
                            X_training=X_initial, y_training=y_initial,
            )
# QUERY strategy 2
elif args.query_strategy == "margin":
    learner = ActiveLearner(estimator=net,
                           query_strategy=modAL.uncertainty.margin_sampling,
                           X_training=X_initial, y_training=y_initial,
                           )
# QUERY strategy 3
elif args.query_strategy == "entropy":
    learner = ActiveLearner(estimator=net,
                            query_strategy=modAL.uncertainty.entropy_sampling,
                            X_training=X_initial, y_training=y_initial,
                            )

prediction_probabilities = []

'''
n_initial is 50% of the labeled ratio
hence we want a total of 2*n_inital number of examples for each labeled ratio from AL to feed into s4gan
target = 2*n_inital

for each query, we are querying 0.1*n_inital samples
query_samples_per_iter = np.floor(0.1*n_inital)


n_queries  = target/query_samples_per_iter
'''
print("n_initial = ", n_initial)
target = math.ceil(float(args.labeled_ratio) * N_total) #10*n_initial # 11 - 0.05, 10 - 0.125
print("target = ", target)
query_samples_per_iter = int(np.ceil(BETA*n_initial))
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

# save the name list and the prediction list:
names_arr = np.array(names[:target])
prediction_prob_arr = np.array(prediction_probabilities[:target])

names_file = os.path.join(dir_name, args.query_strategy + '_names_'+ args.labeled_ratio + '_'+ args.alpha + '_' + args.beta + '.npy')
probs_file = os.path.join(dir_name, args.query_strategy + '_probs_'+ args.labeled_ratio + '_'+ args.alpha + '_' + args.beta + '.npy')

np.save(names_file, names_arr) 
np.save(probs_file, prediction_prob_arr)    
