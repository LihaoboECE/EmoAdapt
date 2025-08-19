# A Portable and Affordable Four-Channel EEG System for Emotion Recognition with Self-Supervised Feature Learning
University of Macau

by Hao Luo, Haobo Li, Wei Tao, Yi Yang, Chio-In Ieong, Feng Wan

# Welcome to EmoAdapt!

## Abstract
Emotions play a pivotal role in shaping human decision-making, behavior, and physiological well-being. Electroencephalography (EEG)-based emotion recognition offers promising avenues for real-time self-monitoring and affective computing applications. However, existing commercial solutions are often hindered by high costs, complicated deployment processes, and limited reliability in practical settings. To address these challenges, we propose a low-cost, self-adaptive wearable EEG system for emotion recognition through a hardware–algorithm co-design approach. The proposed system is a four-channel wireless EEG acquisition device supporting both dry and wet electrodes, with a component cost below USD 35. It features over 7 h of continuous operation, plug-and-play functionality, and modular expandability. At the algorithmic level, we introduce a self-supervised feature extraction framework that combines contrastive learning and masked prediction tasks, enabling robust emotional feature learning from a limited number of EEG channels with constrained signal quality. Our approach attains the highest performance of 60.2% accuracy and 59.4% Macro-F1 score on our proposed platform. Compared to conventional feature-based approaches, it demonstrates a maximum accuracy improvement of up to 20.4% using a multilayer perceptron classifier in our experiment.
**KeyWords: electroencephalogram (EEG); affective computing; few-channel EEG; wearable EEG; self-supervised learning; feature extraction**

## How to run：

### Preparing the environment：

      pip install -r requirements.txt

### Please prepare SEED dataset, you can find here:
https://bcmi.sjtu.edu.cn/home/seed/seed.html

#### Before running, please change the 'dataset_path' in both EmoAdapt_train.py and EmoAdapt_predict.py

### You can download our checkpoint for quick test, please check 
      models/Readme.txt

### EmoAdapt Training:

      demos/EmoAdapt_train.py

### EmoAdapt Predict:

      demos/EmoAdapt_predict.py

