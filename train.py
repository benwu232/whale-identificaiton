# coding: utf-8

# ### **Overview**
# The goal of this competition is identifying individual whales in images. Despite several whales are well represented in images, most of whales are unique or shown only in a few pictures. In particular, the train dataset includes 25k images and 5k unique whale ids. In addition, ~10k of images show unique whales ('new_whale' label). Checking public kernels suggests that a classical approach for classification problems based on softmax prediction for all classes is working quite well for this particular problem. However, strong class imbalance, handling labels represented by just several images, and 'new_whale' label deteriorates this approach. In addition, form the using this model for production, the above approach doesn't sound right since expansion of the model to identify new whales not represented in the train dataset would require retraining the model with increased softmax size. Meanwhile, the task of this competition could be reconsidered as checking similarities that suggests one-shot based learning algorithm to be applicable. This approach is less susceptible to data imbalance in this competition, can naturally handle 'new_whale' class, and is scalable in terms of a model for production (new classes can be added without retraining the model).
# 
# There are several public kernels targeted at using similarity based approach. First of all, it is an amazing [kernel posted by Martin Piotte](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563), which discusses Siamese Neural Network architecture in details. A [fork of this kernel](https://www.kaggle.com/seesee/siamese-pretrained-0-822/notebook) reports 0.822 public LB score after training for 400 epochs. There is also a quite interesting [public kernel](https://www.kaggle.com/ashishpatel26/triplet-loss-network-for-humpback-whale-prediction) discussing Triplet Neural Network architecture, which is supposed to overperform Siamese architecture (check links in [this discussion](https://www.kaggle.com/c/humpback-whale-identification/discussion/76012)). Since both positive and negative examples are provided, the gradients are appeared to be more stable, and the network is not only trying to get away from negative or get close to positive example but arranges the prediction to fulfil both.
# 
# In this kernel I provide an example of a network inspired by Triplet architecture that is capable to reach **~0.81 public LB score after training within the kernel time limit** in my preliminary test. Training for more epochs is supposed to improve the prediction even further. The main trick of this kernel is **using batch all loss**. If the forward pass is completed for all images in a batch, why shouldn't I compare all of them when calculate the loss function? why should I limit myself by just several triplets? I have designed a loss function in such a way that allows performing all vs. all comparison within each batch, in other words for a batch of size 32 instead of comparing 32 triplets or 64 pairs the network performs processing of 9216 pairs of images at the same time. If training is done on multiple GPUs, the number of compared pares could be boosted even further since it it proportional to bs^2. Such a huge number of processed pairs further stabilizes gradients in comparison with triplet loss and allows more effective mapping of the input into the embedding space since not only pairs or triplets but entire picture is seen at the same time. This approach also allows effective search for hard negative examples at the later stage of training since each image is compared with all images in a batch. I tried to boost the search of the most difficult negative examples even further by selection of most similar negative examples to an anchor image when build triplets.
# 
# Moreover, I added [**metric learning**](https://en.wikipedia.org/wiki/Similarity_learning) that boosts the performance of the model really a lot. In my preliminary test **for V2 setup I got 0.606->0.655 improvement** after I started calculating distance as d^2 = (v1-v2).T x A x (v1-v2) instead of Euclidian (v1-v2).T x (v1-v2), where A is a trainable matrix parameter. It can be considered as a trainable deformation of the space. However, the above form of the metric is quite slow at the inference time when distances for all image pairs are calculated. Also, it is quite difficult to impose a constrain on A during training to make it positive semi defined. Therefore, I use an alternative approximation formulation for distance calculation that is much faster at the inference time, symmetric and always positive, and have similar (or slightly better) performance with accounting for nonlinear coordinate transformations. To prevent predictions being spread too much in the embedding space, which deteriorates generalization, I added a compactificcation term to the loss that boosted the score from **0.74 to ~0.771** (V9). When I switched from ResNeXt50 to DenseNet169 backbone **I got 0.771 -> ~0.81 improvement**, and it appeared that DenseNet121 works a little bit better.
# 
# I switched to using cropped rescaled square images since they work better. The idea behind rectangular images generated without distortion of images, which I used in the first versions of the kernel, is the following. Since bounding boxes have different aspect ratio, each image has different degree of distortion when rescaled to square one, which could negatively affect training. However, it looks that the setup when the tail is occupying approximately the same area in the image, no matter what is its orientation and distortion, works better. Looking at the produced images I really do not understand why. In my preliminary test for V7 I could get a **boost from ~0.70 to ~0.75 public LB** after this modification. In the current setup, the images are cropped according to bounding boxes (thanks to [this fork](https://www.kaggle.com/suicaokhoailang/generating-whale-bounding-boxes) and to Martin Piotte for posting the original kernel) and rescaled to 224x224 square images.
# 
# **Milestones of the kernel score improvement and corresponding modifications are summorized in** [**this discussion**](https://www.kaggle.com/c/humpback-whale-identification/discussion/79086#466562). This kernel is written with using fast.ai 0.7 since a newer version of fast.ai doesn't work well in kaggle: using more than one core for data loading leads to [bus error](https://www.kaggle.com/product-feedback/72606) "DataLoader worker (pid 137) is killed by signal: Bus error". Therefore, when I tried to write similar kernel with fast.ai 1.0, it appeared to be much slower, more than 1 hour per epoch vs. 20-30 min with this kernel if ResNet34 and images of size 576x192 are used. People interested in fast.ai 1.0 could check an example of Siamese network [here](https://www.kaggle.com/raghavab1992/siamese-with-fast-ai). Also since fast.ai 0.7 is not really designed to build Siamese and Triplet networks, some parts are a little bit far away from a standard usage of the library.
# 
# **Highlights: Batch all loss, metric learning, mining hard negative examples**

# In[1]:

from fastai.conv_learner import *
from fastai.dataset import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import random
import math
import imgaug as ia
from imgaug import augmenters as iaa
from lib import *

lr = 1e-3
lrs = np.array([lr / 100, lr / 10, lr, lr])

drop_rate = 0.5
fine_tuning = 1

df0 = pd.read_csv(LABELS)
change_new_whale(df0, new_whale_id)

train_list, val_list = split_whale_set(df0, nth_fold=0, new_whale_method=1, seed=1, new_whale_id=new_whale_id)
train_list_nnw, val_list_nnw = split_whale_set(df0, nth_fold=0, new_whale_method=0, seed=1, new_whale_id=new_whale_id)

le = sklearn.preprocessing.LabelEncoder()
le.fit(df0.Id)
df0['label'] = le.transform(df0.Id)

file2label = df0.set_index('Image')

if os.path.isfile('wrong_dict.dump'):
    with open('wrong_dict.dump', 'rb') as f:
        train_set = pickle.load(f)
else:
    train_set = train_list

#trn_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
md, md1 = build_data(sz, bs, train_set, val_list, df0=df0)
'''
x, y = next(iter(md.trn_dl))
print(x.shape, y[0].shape)

to_lb = md.trn_ds.names.Id.to_dict()
#lbs = [[to_lb[idx] for idx in y_cur.tolist()] for y_cur in y]
lbs = [to_lb[idx] for idx in y.tolist()]
# display_imgs((md.trn_ds.denorm(x[:,0,:,:,:]),md.trn_ds.denorm(x[:,1,:,:,:]),              md.trn_ds.denorm(x[:,2,:,:,:])),lbs)
'''


# ### **Training**

# In[21]:

learner = ConvLearner(md, DenseNet121Model(ps=drop_rate, emb_sz=n_embedding))
learner.opt_fn = optim.Adam
learner.clip = 1.0  # gradient clipping
#learner.crit = Contrastive_loss(m=contrastive_neg_margin)
learner.crit = BinaryLoss()
#learner.metrics = [T_acc, BH_acc, pp_dist_max, pn_dist_min]
learner.freeze_to(-2)  # unfreez metric and head block
learner  # click "output" to see details of the model
learner.train_list = train_list
learner.val_list = val_list
learner.df0 = df0
learner.data1 = md1

# First, I train only the fully connected part of the model and the metric while keeping the rest frozen. It allows to avoid corruption of the pretrained weights at the initial stage of training due to random initialization of the head layers. So the power of transfer learning is fully utilized when the training is continued.

# In[22]:
#learner.load('model1')
#learner.half()
#md = get_data(sz, bs, 'train_emb.csv', learner.model)

cb_boost = CbBoost(learner)
if not fine_tuning:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #learner.half()
        learner.fit(lr, 2, callbacks=[cb_boost])
        model_file = 'model0'
        print(f'saving {model_file}')
        learner.save(model_file)
else:
    learner.load('model0')

if 1:
    learner.unfreeze()  # unfreeze entire model
    #learner.half()  # half precision

    # **Since the loss is calculated as an average of nonzero terms, as mentioned above, it's value is not relaiable and must be ignored.** Instead the values of T_acc and BH_acc metrics should be considered.

    # In[ ]:

    n_cycle = 1
    cycle_len = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        learner.fit(lrs / 2, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model1'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 2, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model2'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 4, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model3'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 4, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model4'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 8, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model5'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 8, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model6'
        print(f'saving {model_file}')
        learner.save(model_file)

        n_cycle += 1
        cycle_len += 1
        learner.fit(lrs / 16, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model7'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 16, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model8'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 32, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model9'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 32, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model10'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 64, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model11'
        print(f'saving {model_file}')
        learner.save(model_file)

        learner.fit(lrs / 64, n_cycle, cycle_len=cycle_len, use_clr=(10, 20), callbacks=[cb_boost])
        model_file = 'model12'
        print(f'saving {model_file}')
        learner.save(model_file)


