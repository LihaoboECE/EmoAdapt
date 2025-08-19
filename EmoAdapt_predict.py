'''
    Cross session Test on SEED dataset - based on pura EEG (not DE feature)

    Feature extractor: EmoAdapt (self-supervised model)
    Classifier: SVM (default parameters)

    Author: Lihaobo
    Email: dc22799@umac.mo
    2025/7/3
'''

import os
import torch
import numpy as np
from dataset.seed import SEED
from dataset.paradigms import Emotion

from torch.utils.data import DataLoader
from EmoAdapt import TorchDataset
import argparse
from EmoAdapt import EmoAdapt
from EmoAdapt import plot_embedding
from sklearn.manifold import TSNE
from sklearn import svm
import pandas as pd


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_args(file_name):
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('.', 'models'), type=str)

    parser.add_argument("--host", type=bool, default="127.x.x.1")
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=604)
    return parser.parse_args()

dataset_path = 'E:\SEED'

train_dataset = SEED(path=dataset_path, win_duration=5, sessions=[0])

paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
    # channels=["FP1", "C5", "CP3", "P4"]                            #for 4-ch model
)


X_train, Y_train, _ = paradigm.get_data(
    train_dataset,
    subjects=[i+1 for i in range(15)],
    return_concat=True,
    n_jobs=5,
    verbose=False)


test_dataset = SEED(path=dataset_path, win_duration=5, sessions=[1])

paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
    # channels=["FP1", "C5", "CP3", "P4"]
)

X_test, Y_test, _ = paradigm.get_data(
    test_dataset,
    subjects=[i+1 for i in range(15)],
    return_concat=True,
    n_jobs=5,
    verbose=False)


args = get_args(file_name='EmoAdapt')


# 此模型为自监督模型, 无监督训练
model_path = os.path.join(args.ckpt_path, 'best_model.pth')

model = torch.load(model_path, map_location='cpu', weights_only=False)['model'].to(device)

print("start feature extraction")
# 通过DataLoader节省显存, 也可以使用EmoAdapt.predict()直接预测
train_x, train_y = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
train_dataset = TorchDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

test_x, test_y = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
test_dataset = TorchDataset(test_x, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

# you can try to set disable_BN in False, you will find amazing result (more like TTA or DA)
(latent_train, train_y), (latent_test, test_y) = model.get_latent(train_dataloader, disable_BN=True), model.get_latent(test_dataloader, disable_BN=True)

print("start T-sne")
tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
latent_tsne = tsne.fit_transform(latent_train)
plot_embedding(latent_tsne, train_y, "Session-2 t-SNE")

tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=40)
latent_tsne = tsne.fit_transform(latent_test)
plot_embedding(latent_tsne, test_y, "Session-2 t-SNE")


print("start training")
classifier = svm.SVC()
classifier.fit(latent_train, train_y)
out = classifier.predict(latent_test)
acc = np.mean(out == test_y)
print("ACC: ", acc)

