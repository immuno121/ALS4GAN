# ALS4GAN
This repository contains code for the paper: [Active Learning for Improved Semi Supervised Semantic Segmentation in Satellite Images](https://sites.google.com/view/al-s4gan-semi-sup-sseg/home) which has been accepted at WACV 2022.


Remote sensing data is crucial for applications ranging from monitoring forest fires and deforestation to tracking urbanization. Most of these tasks require dense pixel-level annotations for the model to parse visual information from limited labeled data available for these satellite images. Due to the dearth of high-quality labeled training data in this domain, recent advances have focused on semi-supervised techniques. These techniques generate pseudo-labels from a small set of labeled examples which are used to augment the labeled training set. This makes it necessary to have a highly representative and diverse labeled training set. Therefore, we propose to use an active learning-based sampling strategy to select a highly representative set of labeled training data. We demonstrate our proposed method's effectiveness on two existing semantic segmentation datasets containing satellite images: UC Merced Land Use Classification Dataset, and DeepGlobe Land Cover Classification Dataset. We report a 27% improvement in mIoU with as little as 2% labeled data using active learning sampling strategies over randomly sampling the small set of labeled training data.

![Network Architecture](assets/architecture_diagram.png)

## Pre-requisites
Create a conda enviroment using the provided requirements.yaml file

`conda env create --file=requirements.yaml` 

## Data Peparation

### UCM


### DeepGlobe

## Training and Testing

### Training and Testing on UC Merced Landuse Classification Dataset
  python tools/train_AL.py --dataset-name 'UCM' --query-strategy 'entropy' --random-scale --random-mirror --data-dir '/home/dg777/project/Satellite_Images/' --data-list '/home/dg777/project/Satellite_Images/UCMImageSets/train.txt' --labeled-ratio 0.05  --model-name "res101" --num-classes 21 --learning-rate 0.001 --num-epochs 50 --alpha 0.5 --beta 0.9
  
  python tools/train_AL.py --dataset-name 'deepglobe' --query-strategy 'entropy' --random-scale --random-mirror --data-dir '/home/dg777/project/Satellite_Images/' --data-list '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/train.txt' --labeled-ratio 0.02  --model-name "res50" --num-classes 6 --learning-rate 0.001 --num-epochs 5 --alpha 0.1 --beta 0.9
  
  python tools/train_s4gan.py --dataset deepglobe --labeled-ratio 0.02 --threshold-st 0.8  --num-classes 6 --data-dir '/home/dg777/project/Satellite_Images' --data-list '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/train.txt' --batch-size 10 --num-steps 20000 --restore-from './pretrained_model_checkpoints/deepglobe/checkpoint_final.pth'  --save-pred-every 100 --random-seed 3000
  
  python tools/train_s4gan.py --dataset deepglobe --labeled-ratio 0.02 --threshold-st 0.8  --num-classes 6 --data-dir '/home/dg777/project/Satellite_Images' --data-list '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/train.txt' --batch-size 10 --num-steps 20000 --restore-from './pretrained_model_checkpoints/deepglobe/checkpoint_final.pth'  --save-pred-every 100 --random-seed 3000 --active-learning --active-learning-images-array './active_learning/deepglobe/entropy_names_0.02_0.1_0.9.npy'

python tools/train_s4gan.py --dataset ucm --labeled-ratio 0.02 --threshold-st 0.8  --num-classes 18 --data-dir '/home/dg777/project/Satellite_Images' --data-list '/home/dg777/project/Satellite_Images/UCMImageSets/train.txt' --batch-size 10 --num-steps 20000 --restore-from './pretrained_model_checkpoints/ucm/checkpoint_final.pth'  --save-pred-every 100 --random-seed 3000 --active-learning --active-learning-images-array './active_learning/UCM/entropy_names_0.02_0.1_0.9.npy'
python tools/auto_evaluate.py --start-eval 100 --end-eval 20000 --step-eval 100 --dataset ucm --data-dir '/home/dg777/project/Satellite_Images' --data-list '/home/dg777/project/Satellite_Images/UCMImageSets/test.txt' --num-classes 18 --checkpoint-dir './checkpoint/' --input-size '320,320'
 
python tools/auto_evaluate.py --start-eval 100 --end-eval 20000 --step-eval 100 --dataset deepglobe --data-dir '/home/dg777/project/Satellite_Images' --data-list '/home/dg777/project/Satellite_Images/DeepGlobeImageSets/test.txt' --num-classes 6 --checkpoint-dir './checkpoint/' --input-size '320,320'

###  Training and Testing on DeepGlobe LandCover Dataset

## Acknowledgements

## Citation
