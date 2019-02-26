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
from fastai.sgdr import Callback


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

# In[ ]:

PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
BOXES = '../input/bounding_boxes.csv'
# MODLE_INIT = '../input/pytorch-pretrained-models/'

contrastive_neg_margin = 10.0
n_embedding = 512
#bs = 32
bs = 5
bs = 26
sz = 384 # increase the image size at the later stage of training
sz = 224  # increase the image size at the later stage of training
nw = 6

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

new_whale_id = 'z_new_whale'

import datetime as dt
def now2str(format="%Y-%m-%d_%H-%M-%S"):
    #str_time = time.strftime("%Y-%b-%d-%H-%M-%S", time.localtime(time.time()))
    return dt.datetime.now().strftime(format)

def change_new_whale(df, new_name='z_new_whale'):
    for k in range(len(df)):
        if df.at[k, 'Id'] == 'new_whale':
            df.at[k, 'Id'] = new_name

def split_whale_set(df, nth_fold=0, total_folds=5, new_whale_method=0, seed=1, new_whale_id='z_new_whale'):
    '''
    Split whale dataset to train and valid set based on k-fold idea.
    total_folds: number of total folds
    nth_fold: the nth fold
    new_whale_method: If 0, remove new_whale in all data sets; if 1, add new_whale to train/validation sets
    seed: Random seed for shuffling
    '''
    np.random.seed(seed)
    # list(df_known.groupby('Id'))
    train_list = []
    val_list = []
    # df_known = df[df.Id!='new_whale']
    for name, group in df.groupby('Id'):
        # print(name, len(group), group.index, type(group))
        # if name == 'w_b82d0eb':
        #    print(name, df_known[df_known.Id==name])
        if new_whale_method == 0 and name == new_whale_id:
            continue
        group_num = len(group)
        images = group.Image.values
        if group_num > 1:
            np.random.shuffle(images)
            # images = list(images)
            span = max(1, group_num // total_folds)
            val_images = images[nth_fold * span:(nth_fold + 1) * span]
            train_images = list(set(images) - set(val_images))
            val_list.extend(val_images)
            train_list.extend(train_images)
        else:
            train_list.extend(images)

    return train_list, val_list


# ### **Data**
# The class Loader creates crops with sizes 224x224 based on the bounding boxes. In addition, data augmentation based on [imgaug library](https://github.com/aleju/imgaug) is applied. This library is quite interesting in the context of the competition since it supports hue and saturation augmentations as well as conversion to gray scale. Following the idea of progressive resizing on the later stage of training one can switch to higher resolution images, but at the beginning low resolution is used to speed up the convergence and improve generalization ability of the model. 

# In[10]:

def open_image(fn):
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags)
            if im is None:
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


class Loader():
    def __init__(self, path, tfms_g=None, tfms_px=None):
        # tfms_g - geometric augmentation (distortion, small rotation, zoom)
        # tfms_px - pixel augmentation and flip
        self.boxes = pd.read_csv(BOXES).set_index('Image')
        self.path = path
        self.tfms_g = iaa.Sequential(tfms_g, random_order=False) if tfms_g is not None else None
        self.tfms_px = iaa.Sequential(tfms_px, random_order=False) if tfms_px is not None else None

    def __call__(self, fname):
        fname = os.path.basename(fname)
        x0, y0, x1, y1 = tuple(self.boxes.loc[fname, ['x0', 'y0', 'x1', 'y1']].tolist())
        img = open_image(os.path.join(self.path, fname))
        l1, l0, _ = img.shape
        b0, b1 = x1 - x0, y1 - y0
        # padding
        x0n, x1n = max(int(x0 - b0 * 0.05), 0), min(int(x1 + b0 * 0.05), l0 - 1)
        y0n, y1n = max(int(y0 - b1 * 0.05), 0), min(int(y1 + b1 * 0.05), l1 - 1)

        if self.tfms_g != None: img = self.tfms_g.augment_image(img)
        img = cv2.resize(img[y0n:y1n, x0n:x1n, :], (sz, sz))
        if self.tfms_px != None: img = self.tfms_px.augment_image(img)
        return img.astype(np.float) / 255


# The pdFilesDataset class below generates triplets of images: original image, different image with the same label, an image with different label (**including new_label images**). Image_selection class performs of selection of the most difficult negative examples used in the last part of the kernel. When this class is created, 64 most similar images with different label are selected for each image. So instead of random sampling of negative examples, sampling of these 64 most difficult images for each anchor image can be used during training.

# In[11]:

def get_idxs0(names, df, n=64):
    idxs = []
    for name in names:
        label = df[df.Image == name].Id.values[0]
        idxs.append(df[df.Id != label].Image.sample(n).values)
    return idxs


class Image_selection:
    def __init__(self, fnames, df0, emb_df=None, model=None):
        if emb_df is None or model is None:
            df = df0[df0.Image]
            '''
            df = data.copy()  # .set_index('Image')
            counts = Counter(df.Id.values)
            df['c'] = df['Id'].apply(lambda x: counts[x])
            df['label'] = df.Id
            df.loc[df.c == 1, 'label'] = 'new_whale'
            df = df.sort_values(by=['c'])
            df.label = pd.factorize(df.label)[0]
            l1 = 1 + df.label.max()
            l2 = len(df[df.label == 0])
            df.loc[df.label == 0, 'label'] = range(l1, l1 + l2)  # assign unique ids
            df = df.set_index('label')
            '''
            df_label = df0.set_index('label')
            l = len(fnames)
            idxs = Parallel(n_jobs=nw)(delayed(get_idxs0)(fnames[int(i * l / nw):int((i + 1) * l / nw)], df_label) for i in range(nw))
            idxs = [y for x in idxs for y in x]
            pass
        else:
            data = df0.copy().set_index('Image')
            trn_emb = emb_df.copy()
            trn_emb.set_index('files', inplace=True)
            trn_emb['emb'] = [[float(i) for i in s.split()] for s in trn_emb['emb']]
            trn_emb = data.join(trn_emb)
            trn_emb = trn_emb.reset_index()
            trn_emb['idx'] = np.arange(len(trn_emb))
            trn_emb = trn_emb.set_index('Id')
            emb = np.array(trn_emb.emb.tolist())
            l = len(fnames)
            idxs = []
            sort_list = []
            model.eval()
            with torch.no_grad():
                # selection of the most difficult negative examples
                m = model.module if isinstance(model, FP16) else model
                emb = torch.from_numpy(emb).half().cuda()
                for name in tqdm(fnames):
                    label = trn_emb.loc[trn_emb.Image == name].index[0]
                    v0 = np.array(trn_emb.loc[trn_emb.Image == name, 'emb'].tolist()[0])
                    v0 = torch.from_numpy(v0).half().cuda()
                    d = m.get_d(v0, emb)
                    ids = trn_emb.loc[trn_emb.index != label].idx.tolist()
                    #sorted, indices = torch.sort(d[ids])
                    sorted, indices = torch.topk(d[ids], 64, largest=False)
                    sort_list.append(sorted)
                    idxs.append([ids[i] for i in indices[:64]])
            trn_emb = trn_emb.set_index('idx')
            idxs = [trn_emb.loc[idx, 'Image'] for idx in idxs]
        self.df = pd.DataFrame({'Image': fnames, 'idxs': idxs}).set_index('Image')

    def get(self, name):
        return np.random.choice(self.df.loc[name].values[0], 1)[0]


# In[12]:

class pdFilesDataset(FilesDataset):
    def __init__(self, data, path, transform, df0):
        df = df0.copy()
        self.fnames_dict = data
        self.fnames_list = data
        self.is_dict = isinstance(data, dict)
        if self.is_dict:
            self.fnames_list = list(data.keys())

        '''
        counts = Counter(df.Id.values)
        df['c'] = df['Id'].apply(lambda x: counts[x])
        # in the production runs df.c>1 should be used
        fnames = df[(df.c > 2) & (df.Id != 'new_whale')].Image.tolist()
        df['label'] = df.Id
        df.loc[df.c == 1, 'label'] = 'new_whale'
        df = df.sort_values(by=['c'])
        df.label = pd.factorize(df.label)[0]
        l1 = 1 + df.label.max()
        l2 = len(df[df.label == 0])
        df.loc[df.label == 0, 'label'] = range(l1, l1 + l2)  # assign unique ids
        '''
        self.labels = df.copy().set_index('Image')
        self.names = df.copy().set_index('label')
        if path == TRAIN:
            # data augmentation: 8 degree rotation, 10% stratch, shear
            tfms_g = [iaa.Affine(rotate=(-8, 8), mode='reflect',
                                 scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, shear=(-16, 16))]
            # data augmentation: horizontal flip, hue and staturation augmentation,
            # gray scale, blur
            tfms_px = [iaa.Fliplr(0.5), iaa.AddToHueAndSaturation((-20, 20)),
                       iaa.Grayscale(alpha=(0.0, 1.0)), iaa.GaussianBlur((0, 1.0))]
            self.loader = Loader(path, tfms_g, tfms_px)
        else:
            self.loader = Loader(path)
        self.selection = None
        super().__init__(self.fnames_list, transform, path)

    def get_x(self, i):
        while True:
            label = self.labels.loc[self.fnames_list[i], 'label']
            if label != 5004:
                break
            else:
                i = np.random.randint(len(self.fnames_list))
        # random selection of a positive example
        for j in range(10):  # sometimes loc call fails
            try:
                names = self.names.loc[label].Image
                break
            except:
                None
        name_p = names if isinstance(names, str) else random.sample(set(names) - set([self.fnames_list[i]]), 1)[0]

        # random selection of a negative example
        if not self.is_dict:
            for j in range(10):  # sometimes loc call fails
                try:
                    names = self.names.loc[self.names.index != label].Image
                    break
                except:
                    names = self.fnames_list[i]
            name_n = names if isinstance(names, str) else names.sample(1).values[0]
        else:
            name_n = random.choice(self.fnames_dict[self.fnames_list[i]])
        imgs = [self.loader(os.path.join(self.path, self.fnames_list[i])),
                self.loader(os.path.join(self.path, name_p)),
                self.loader(os.path.join(self.path, name_n)),
                label, label, self.labels.loc[name_n, 'label']]
        return imgs

    def get_y(self, i):
        return 0

    def get(self, tfm, x, y):
        if tfm is None:
            return (*x, 0)
        else:
            x1, y1 = tfm(x[0], x[3])
            x2, y2 = tfm(x[1], x[4])
            x3, y3 = tfm(x[2], x[5])
            # combine all images into one tensor
            x = np.stack((x1, x2, x3), 0)
            return x, (y1, y2, y3)

    def get_names(self, label):
        names = []
        for j in range(10):
            try:
                names = self.names.loc[label].Image
                break
            except:
                None
        return names

    @property
    def is_multi(self):
        return True

    @property
    def is_reg(self):
        return True

    def get_c(self):
        return n_embedding

    def get_n(self):
        return len(self.fnames_list)


# class for loading an individual images when embedding is computed
class FilesDataset_single(FilesDataset):
    def __init__(self, data, path, transform):
        self.loader = Loader(path)
        #fnames = os.listdir(path)
        fnames = sorted(list(Path(path).glob('*.jpg')))
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        return self.loader(os.path.join(self.path, self.fnames[i]))

    def get_y(self, i):
        return 0

    @property
    def is_multi(self): return True

    @property
    def is_reg(self): return True

    def get_c(self): return n_embedding

    def get_n(self): return len(self.fnames)


class FileDs(FilesDataset):
    def __init__(self, data, path, transform, df0):
        self.df0 = df0
        self.fnames_dict = data
        self.fnames_list = data
        self.is_dict = isinstance(data, dict)
        if self.is_dict:
            self.fnames_list = list(data.keys())
        self.labels = df0.set_index('Image')
        if path == TRAIN:
            # data augmentation: 8 degree rotation, 10% stratch, shear
            tfms_g = [iaa.Affine(rotate=(-8, 8), mode='reflect',
                                 scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, shear=(-16, 16))]
            # data augmentation: horizontal flip, hue and staturation augmentation,
            # gray scale, blur
            tfms_px = [iaa.Fliplr(0.5), iaa.AddToHueAndSaturation((-20, 20)),
                       iaa.Grayscale(alpha=(0.0, 1.0)), iaa.GaussianBlur((0, 1.0))]
            self.loader = Loader(path, tfms_g, tfms_px)
        else:
            self.loader = Loader(path)
        self.selection = None
        super().__init__(self.fnames_list, transform, path)

    def get_x(self, i):
        label = self.labels.loc[self.fnames_list[i], 'label']
        imgs = [self.loader(os.path.join(self.path, self.fnames_list[i])), label]
        return imgs

    def get_y(self, i):
        return 0

    def get(self, tfm, x, y):
        if tfm is None:
            return (*x, 0)
        else:
            x1, y1 = tfm(x[0], x[1])
            return x1, y1

    def get_names(self, label):
        names = []
        for j in range(10):
            try:
                names = self.names.loc[label].Image
                break
            except:
                None
        return names

    @property
    def is_multi(self):
        return True

    @property
    def is_reg(self):
        return True

    def get_c(self):
        return n_embedding

    def get_n(self):
        return len(self.fnames_list)


class FileDs1(FilesDataset):
    def __init__(self, data, path, transform):
        self.loader = Loader(path)
        #fnames = os.listdir(path)
        if data is None:
            fnames = sorted(list(Path(path).glob('*.jpg')))
        elif isinstance(data, pd.DataFrame):
            fnames = sorted(data.Image.to_list())
        elif isinstance(data, list):
            fnames = sorted(data)
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        return self.loader(os.path.join(self.path, self.fnames[i]))

    def get_y(self, i):
        return 0

    @property
    def is_multi(self): return True

    @property
    def is_reg(self): return True

    def get_c(self): return n_embedding

    def get_n(self): return len(self.fnames)

# In[13]:

def get_data(sz, bs, fname_emb=None, model=None):
    tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO)
    tfms[0].tfms = [tfms[0].tfms[2], tfms[0].tfms[3]]
    tfms[1].tfms = [tfms[1].tfms[2], tfms[1].tfms[3]]
    df = pd.read_csv(LABELS)
    trn_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    ds = ImageData.get_ds(pdFilesDataset, (trn_df, TRAIN), (val_df, TRAIN), tfms)
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    if fname_emb != None and model != None:
        print('selecting samples')
        emb = pd.read_csv(fname_emb)
        md.trn_dl.dataset.selection = Image_selection(md.trn_dl.dataset.fnames, trn_df, emb, model)
        md.val_dl.dataset.selection = Image_selection(md.val_dl.dataset.fnames, val_df, emb, model)
    return md


def build_data(sz, bs, train_set, val_set, df0, fname_emb=None, model=None):
    tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO)
    tfms[0].tfms = [tfms[0].tfms[2], tfms[0].tfms[3]]
    tfms[1].tfms = [tfms[1].tfms[2], tfms[1].tfms[3]]
    #ds = ImageData.get_ds(FileDs1, (trn_df, TRAIN), (val_df, TRAIN), tfms, test=(None, TEST))
    #md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)

    ds = ImageData.get_ds(pdFilesDataset, (train_set, TRAIN), (val_set, TRAIN), tfms, df0=df0)
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)

    ds1 = ImageData.get_ds(FileDs, (train_set, TRAIN), (val_set, TRAIN), tfms, df0=df0)#, test=(None, TEST))
    md1 = ImageData(PATH, ds1, bs, num_workers=nw, classes=None)

    '''
    if fname_emb != None and model != None:
        print('selecting samples')
        emb = pd.read_csv(fname_emb)
        md.trn_dl.dataset.selection = Image_selection(train_set, df0, emb, model)
        md.val_dl.dataset.selection = Image_selection(val_set, df0, emb, model)
    '''
    return md, md1


def idx2label(ds):
    labels = []
    for k, fname in enumerate(ds.fnames):
        labels.append(ds.labels.loc[fname, 'label'])
    return labels


class CbBoost(Callback):
    def __init__(self, learn):
        super().__init__()
        self.learn = learn
        self.train_labels = idx2label(self.learn.data1.trn_ds)
        self.val_labels = idx2label(self.learn.data1.val_ds)

    #def on_epoch_end(self, none) -> None:
    def on_batch_end(self, none) -> None:
        print('************************************* Boosting ****************************************')
        print(now2str())
        print('Calculating map5 in validation set ...')
        if 1:
            met_mat = cal_ds_metrics(self.learn.model, self.learn.data1.val_dl)
        else:
            with open('metmat.dump', 'rb') as f:
                met_mat = pickle.load(f)
        map5 = cal_map_score(met_mat, self.val_labels, threshold=0.7)
        print(f'Validation set, map5 = {map5}')

        print(now2str())
        print('Finding wrong images in train set ... ')
        if 1:
            met_mat = cal_ds_metrics(self.learn.model, self.learn.data1.trn_dl)
            #with open('metmat_trn.dump', 'wb') as f:
            #    pickle.dump(met_mat, f)
        else:
            with open('metmat_trn.dump', 'rb') as f:
                met_mat = pickle.load(f)
        wrong_dict = find_wrong_files(met_mat, self.learn.data1.trn_ds, self.train_labels)
        self.learn.data_, _ = build_data(sz, bs, wrong_dict, self.learn.val_list, df0=self.learn.df0)
        print(now2str())
        print(f'Wrong dict size {len(wrong_dict)}.')
        with open('wrong_dict.dump', 'wb') as f:
            pickle.dump(wrong_dict, f)
        print('\n')



def cal_ds_metrics(model, dl):
    emb, names = cal_emb(model, dl)
    emb_size = emb.shape[0]
    metric_list = []
    model.eval()
    with torch.no_grad():
        m = model.module if isinstance(model, FP16) else model
        for k in (range(emb_size)):
            emb_row = emb[k].expand((*(emb.shape)))
            metric = m.get_d(emb_row, emb)
            metric_list.append(metric.view(-1))
    metrics = torch.stack(metric_list)
    return metrics

def cal_map_score(met_mat, labels, mapn=5, threshold=0.7):
    new_whale_label = 5004
    sorted, indices = torch.topk(met_mat, 64, dim=1, largest=False)
    col_len, row_len = met_mat.shape
    scores = np.zeros(col_len)
    for idx in range(col_len):
        ref_label = labels[idx]
        cnt = 0
        score = 0.0
        for k in range(row_len):
            cur_k = indices[idx, k]
            cur_value = sorted[idx, k]
            if cur_k == idx:
                continue

            cur_label = labels[cur_k]
            if cur_value > threshold:
                cur_label = new_whale_label

            if ref_label == cur_label:
                score = 1.0 / (cnt + 1)
                break
            else:
                cnt += 1
                if cnt >= mapn:
                    break
        scores[idx] = score
    return scores.mean()

def find_wrong_files(met_mat, ds, labels):
    #metrics = cal_ds_metrics(model, dl)
    sorted, indices = torch.topk(met_mat, 64, dim=1, largest=False)
    col_len, row_len = sorted.shape
    wrong_idxes = []
    wrong_dict = {}
    for idx in range(col_len):
        ref_label = labels[idx]
        for k in range(row_len):
            cur_k = indices[idx, k]
            if cur_k == idx:
                continue

            cur_label = labels[cur_k]
            if cur_label != ref_label:
                #print(idx, ds.fnames_list[idx])
                if ds.fnames_list[idx] not in wrong_dict:
                    wrong_dict[ds.fnames_list[idx]] = [ds.fnames[cur_k.item()]]
                else:
                    wrong_dict[ds.fnames_list[idx]].append(ds.fnames[cur_k.item()])
            else:
                break
    return wrong_dict

    '''
    #emb, names = cal_emb(self.learn.model, self.learn.data.trn_dl)
    wrong_idxes = list(set(wrong_idxes))
    wrong_fnames = []
    for idx in wrong_idxes:
        #print(len(wrong_idxes), k)
        wrong_fnames.append(ds.fnames[idx])

    return wrong_fnames
    '''


# The image below demonstrates an example of triplets of rectangular 576x192 augmented images used for training. To be honest, some of those triplets are quite hard, and I don't think that I could even reach the same performance as the model after training (~99% accuracy in identifications of 2 similar images in a triplet).

# In[14]:


def display_imgs(x, lbs=None):
    columns = 3
    rows = min(bs, 16)
    ig, ax = plt.subplots(rows, columns, figsize=(columns * 5, rows * 5))
    for i in range(rows):
        for j in range(columns):
            idx = j + i * columns
            ax[i, j].imshow((x[j][i, :, :, :] * 255).astype(np.int))
            ax[i, j].axis('off')
            if lbs is not None:
                ax[i, j].text(10, 25, lbs[j][i], size=20, color='red')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


# ### **Model**
# Looking to the images I would expect that larger models should perform better in this competition since there are many really difficult examples that require perception capability surpassing human ones (check examples in the last part of the kernel). The convolutional part is taken from the original DenseNet121 model pretrained on ImageNet, and adaptive pooling allows using of images of any sizes and aspect ratios. The convolutional part is followed by 2 fully connected layers added to convert the prediction into embedding space.
# 
# On the top, a learnable metric is applied that converts differences of vector components in the embedding space for all images within a batch into distances. This part replaces calculation of distances in Euclidean space by one in distorted space. The first thing that came me in mind is generalized formula for a distance in a linear space d^2 = (v1-v2).T x A x (v1-v2), where A is a trainable matrix. For Euclidian space d^2 = (v1-v2).T x (v1-v2), i.e. A is a identity matrix. Using of such distance boosted the score of the v2 version of this kernel from 0.606 to 0.655. However, the above form of the metric is quite slow at the inference time when distances for all image pairs are calculated. Also, it is quite difficult to impose a constrain on A during training to make it positive semi defined (distance is always positive). Therefore, I use an alternative approximation formulation for distance calculation that is much faster at the inference time, symmetric and always positive, and have similar (or slightly better) performance. If after calculation of differences between v1 and v2 the result goes through a linear layer, before summing the squares, the distance would include also contribution from mixed terms like dv1 * dv2 (like in an approach with matrix A), though many of these terms are correlated, and results are a little bit worse (it can be viewed as a factorization of the general approach). However, when I included also (v1-v2)^2 terms as an input, the performance of such metric surpassed one for d^2 = (v1-v2).T x A x (v1-v2). It can be explained as an approximation of some nonlinear transformation of a space including square, cubic, and quadratic terms. The concept of metric learning is similar to one in [Martin's kernel](https://www.kaggle.com/martinpiotte/whale-recognition-model-with-score-0-78563). However, in that case the metric converts the difference in the probability of two classes to have the same label, while in the current kernel it converts the difference between vectors into a distance in a distorted space.

# In[15]:

class Metric(nn.Module):
    def __init__(self, emb_sz=64):
        super().__init__()
        self.l = nn.Linear(emb_sz * 2, emb_sz * 2, False)
        self.fc = nn.Linear(emb_sz * 2, 1)

    def forward(self, d):
        d2 = d.pow(2)
        d = self.l(torch.cat((d, d2), dim=-1))
        logits = self.fc(d)
        return torch.sigmoid(logits)


# In[16]:

def resnext50(pretrained=True):
    model = resnext_50_32x4d()
    name = 'resnext_50_32x4d.pth'
    if pretrained:
        path = os.path.join(MODLE_INIT, name)
        load_model(model, path)
    return model


class TripletResneXt50(nn.Module):
    def __init__(self, pre=True, emb_sz=64, ps=0.5):
        super().__init__()
        encoder = resnext50(pretrained=pre)
        # add DataParallel to allow support of multiple GPUs
        self.cnn = nn.DataParallel(nn.Sequential(encoder[0], encoder[1], nn.ReLU(),
                                                 encoder[3], encoder[4], encoder[5], encoder[6], encoder[7]))
        self.head = nn.DataParallel(nn.Sequential(AdaptiveConcatPool2d(), Flatten(),
                                                  nn.Dropout(ps), nn.Linear(4096, 512), nn.ReLU(),
                                                  nn.BatchNorm1d(512), nn.Dropout(ps), nn.Linear(512, emb_sz)))
        self.metric = nn.DataParallel(Metric(emb_sz))

    def forward(self, x):
        x1, x2, x3 = x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :]
        x1 = self.head(self.cnn(x1))
        x2 = self.head(self.cnn(x2))
        x3 = self.head(self.cnn(x3))
        x = torch.cat((x1, x2, x3))
        sz = x.shape[0]
        x1 = x.unsqueeze(1).expand((sz, sz, -1))
        x2 = x1.transpose(0, 1)
        # matrix of all vs all differencies
        d = (x1 - x2).view(sz * sz, -1)
        return self.metric(d)

    def get_embedding(self, x):
        return self.head(self.cnn(x))

    def get_d(self, x0, x):
        d = (x - x0)
        return self.metric(d)


class ResNeXt50Model():
    def __init__(self, pre=True, name='TripletResneXt50', **kwargs):
        self.model = to_gpu(TripletResneXt50(pre=True, **kwargs))
        self.name = name

    def get_layer_groups(self, precompute):
        m = self.model.module if isinstance(self.model, FP16) else self.model
        if precompute:
            return [m.head] + [m.metric]
        c = children(m.cnn.module)
        return list(split_by_idxs(c, [5])) + [m.head] + [m.metric]


# In[17]:

def get_densenet169(pre=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = cut_model(cut_model(densenet169(pre), -1)[0], -1)
    return nn.Sequential(*layers)


class TripletDenseNet169(nn.Module):
    def __init__(self, pre=True, emb_sz=64, ps=0.5):
        super().__init__()
        encoder = get_densenet169(pre)
        # add DataParallel to allow support of multiple GPUs
        self.cnn = nn.DataParallel(nn.Sequential(encoder[0], encoder[1], nn.ReLU(),
                                                 encoder[3], encoder[4], encoder[5], encoder[6], encoder[7],
                                                 encoder[8], encoder[9], encoder[10]))
        self.head = nn.DataParallel(nn.Sequential(AdaptiveConcatPool2d(), Flatten(),
                                                  nn.Dropout(ps), nn.Linear(3328, 512), nn.ReLU(),
                                                  nn.BatchNorm1d(512), nn.Dropout(ps), nn.Linear(512, emb_sz)))
        self.metric = nn.DataParallel(Metric(emb_sz))

    def forward(self, x):
        x1, x2, x3 = x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :]
        x1 = self.head(self.cnn(x1))
        x2 = self.head(self.cnn(x2))
        x3 = self.head(self.cnn(x3))
        x = torch.cat((x1, x2, x3))
        sz = x.shape[0]
        x1 = x.unsqueeze(1).expand((sz, sz, -1))
        x2 = x1.transpose(0, 1)
        # matrix of all vs all differencies
        d = (x1 - x2).view(sz * sz, -1)
        return self.metric(d)

    def get_embedding(self, x):
        return self.head(self.cnn(x))

    def get_d(self, x0, x):
        d = (x - x0)
        return self.metric(d)


class DenseNet169Model():
    def __init__(self, pre=True, name='TripletDenseNet169', **kwargs):
        self.model = to_gpu(TripletDenseNet169(pre=True, **kwargs))
        self.name = name

    def get_layer_groups(self, precompute):
        m = self.model.module if isinstance(self.model, FP16) else self.model
        if precompute:
            return [m.head] + [m.metric]
        c = children(m.cnn.module)
        return list(split_by_idxs(c, [8])) + [m.head] + [m.metric]


# In[18]:

def get_densenet121(pre=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layers = cut_model(cut_model(densenet121(pre), -1)[0], -1)
    return nn.Sequential(*layers)


class TripletDenseNet121(nn.Module):
    def __init__(self, pre=True, emb_sz=64, ps=0.5):
        super().__init__()
        encoder = get_densenet121(pre)
        # add DataParallel to allow support of multiple GPUs
        self.cnn = nn.DataParallel(nn.Sequential(encoder[0], encoder[1], nn.ReLU(),
                                                 encoder[3], encoder[4], encoder[5], encoder[6], encoder[7],
                                                 encoder[8], encoder[9], encoder[10]))
        self.head = nn.DataParallel(nn.Sequential(AdaptiveConcatPool2d(), Flatten(),
                                                  nn.BatchNorm1d(2048), nn.Dropout(ps), nn.Linear(2048, 1024),
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(1024), nn.Dropout(ps), nn.Linear(1024, emb_sz)))
        self.metric = nn.DataParallel(Metric(emb_sz))
        self.n_slice = 16

    def forward(self, x):
        z = x.view(-1, *x.shape[2:])
        x = self.head(self.cnn(z))

        #x1, x2, x3 = x[:, 0, :, :, :], x[:, 1, :, :, :], x[:, 2, :, :, :]
        #x1 = self.head(self.cnn(x1))
        #x2 = self.head(self.cnn(x2))
        #x3 = self.head(self.cnn(x3))
        #x = torch.cat((x1, x2, x3))
        sz = x.shape[0]
        x1 = x.unsqueeze(1).expand((sz, sz, -1))
        x2 = x1.transpose(0, 1)
        # matrix of all vs all differencies
        d = (x1 - x2).view(sz * sz, -1)
        return self.metric(d)

    def get_embedding(self, x):
        return self.head(self.cnn(x))

    def get_d(self, x0, x):
        d = (x - x0)
        return self.metric(d)


class DenseNet121Model():
    def __init__(self, pre=True, name='TripletDenseNet21', **kwargs):
        self.model = to_gpu(TripletDenseNet121(pre=True, **kwargs))
        self.name = name

    def get_layer_groups(self, precompute):
        m = self.model.module if isinstance(self.model, FP16) else self.model
        if precompute:
            return [m.head] + [m.metric]
        c = children(m.cnn.module)
        return list(split_by_idxs(c, [8])) + [m.head] + [m.metric]


# ### **Loss function**
# I my tests I have performed a comparison of several loss functions and found that contrastive loss works the best in the current setup. I also, in comparison with the previous version, select only nonzero terms when perform averaging. I found that at a later stage of training many pairs are already well separated and contribute zero to gradients while decrease the magnitude of useful gradients during averaging. The drawback of such an approach is that during training **values of train and validation loss must be ignored, since they are calculated each time based on different number of pairs**, and only the value of metric (like accuracy of identifying a correct pair of images with the same label in a triplet, T_acc, or accuracy of identifying a correct pair of images with the same label within a hardest triplet in batch for each anchor image, BH_acc) must be tracked. However, after performing mining hardest negative examples, even accuracy metrics are not reliable, and the only way to check performance of the model is validation based on the entire validation dataset rather than batches with using the same metric as one in the competition. To prevent predictions being spread too much in the embedding space, which deteriorates generalization, in the V9 of the kernel I added a compactificcation term to the loss that boosted the score from **0.74 to ~0.78** and stopped unstable behaviour of val loss.

# In[19]:

class BinaryLoss(nn.Module):
    def __init__(self, wd=1e-4):
        super().__init__()
        self.wd =  wd

    def forward(self, d, target):
        if isinstance(target, list):
            target = torch.cat(target)
        #d = d.float()
        d = d.view(-1)
        # matrix of all vs all comparisons
        #t = torch.cat(target)
        t = target
        sz = t.shape[0]
        t1 = t.unsqueeze(1).expand((sz, sz))
        t2 = t1.transpose(0, 1)
        y = ((t1 != t2) + to_gpu(torch.eye(sz).byte())).view(-1)
        y = y.half()
        #y = y.float()

        loss_p = None
        if len(y[y==0]):
            loss_p = F.binary_cross_entropy(d[y==0], y[y==0])

        loss_n = None
        if len(y[y==1]):
            loss_n = F.binary_cross_entropy(d[y==1], y[y==1])

        if loss_p is None:
            return loss_n

        loss = (loss_p + loss_n) / 2
        return loss


class Contrastive_loss(nn.Module):
    def __init__(self, m=10.0, wd=1e-4):
        super().__init__()
        self.m, self.wd = m, wd

    def forward(self, d, target):
        d = d.float()
        # matrix of all vs all comparisons
        t = torch.cat(target)
        sz = t.shape[0]
        t1 = t.unsqueeze(1).expand((sz, sz))
        t2 = t1.transpose(0, 1)
        y = ((t1 == t2) + to_gpu(torch.eye(sz).byte())).view(-1)

        loss_p = d[y == 1]
        loss_n = F.relu(self.m - torch.sqrt(d[y == 0])) ** 2
        loss = torch.cat((loss_p, loss_n), 0)
        loss = loss[torch.nonzero(loss).squeeze()]
        loss = loss.mean() if loss.shape[0] > 0 else loss.sum()
        loss += self.wd * (d ** 2).mean()  # compactification term
        return loss


# accuracy within a triplet
def T_acc(d, target):
    sz = target[0].shape[0]
    lp = [3 * sz * i + i + sz for i in range(sz)]
    ln = [3 * sz * i + i + 2 * sz for i in range(sz)]
    dp, dn = d[lp], d[ln]
    return (dp < dn).float().mean()


# accuracy within a hardest triplet in a batch for each anchor image
def BH_acc(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    BH = []
    for i in range(sz):
        dp = d[i, y[i, :] == 1].max()
        dn = d[i, y[i, :] == 0].min()
        BH.append(dp < dn)
    return torch.FloatTensor(BH).float().mean()


def pp_dist_max(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    pp_dist = []
    for i in range(sz):
        dp = d[i, y[i] == 1].max()
        dn = d[i, y[i] == 0].min()
        pp_dist.append(dp)
    return torch.FloatTensor(pp_dist).float().mean()

def pn_dist_min(d, target):
    t = torch.cat(target)
    sz = t.shape[0]
    t1 = t.unsqueeze(1).expand((sz, sz))
    t2 = t1.transpose(0, 1)
    y = (t1 == t2)
    d = d.float().view(sz, sz)
    pn_dist = []
    for i in range(sz):
        dn = d[i, y[i] == 0].min()
        pn_dist.append(dn)
    return torch.FloatTensor(pn_dist).float().mean()


# I also tried batch hard triplet loss introduced in https://arxiv.org/pdf/1703.07737.pdf , however I didn't get better results when my current one yet. The problem, I expect, is that triplet loss doesn't try to bring all predictions with the same label to one point in an embedding space, while keeping objects of the same kind together. If would be great if there was no new_whale class. The check I perform is based on the distance to the nearest neighbors: if there are no whales in a sphere of a particular radius, I assign new whale label. It works well if the points are packed with approximately the same density, but if for some classes the spacing between images of the same class is too large, they will be misclassified as a new_whale.
# 
# The class in the hidden cell below implements such approach. The thing I have introduced here is a parameter n that indicates how many hardest triplets should be selected per batch, which allows gradually going from batch all to batch hard loss during training, when the fraction of hard triplets drops (though, sorting takes additional time).

# In[20]:

# batch hard loss: https://arxiv.org/pdf/1703.07737.pdf
class BH_loss(nn.Module):
    def __init__(self, n=1, m=0.0):
        super().__init__()
        self.n = n  # select n hardest triplets
        self.m = m

    def forward(self, input, target, size_average=True):
        # matrix of all vs all comparisons
        t = torch.cat(target)
        sz = t.shape[0]
        t1 = t.unsqueeze(1).expand((sz, sz))
        t2 = t1.transpose(0, 1)
        y = ((t1 == t2) + to_gpu(torch.eye(sz).byte()))
        d = input.float().view(sz, sz)
        D = []
        for i in range(2 * sz // 3):
            dp = d[i, y[i, :] == 1].max()
            dn = d[i, y[i, :] == 0]
            dist, idxs = torch.sort(dn)
            n = min(self.n, dn.shape[0])
            for j in range(n):
                D.append((self.m + dp - dn[idxs[j]]).unsqueeze(0))
        loss = torch.log1p(torch.exp(torch.cat(D)))
        loss = loss.mean() if size_average else loss.sum()
        return loss


# First, I train only the fully connected part of the model and the metric while keeping the rest frozen. It allows to avoid corruption of the pretrained weights at the initial stage of training due to random initialization of the head layers. So the power of transfer learning is fully utilized when the training is continued.

# In[22]:


# Next, I unfreeze all weights and allow training of entire model. One trick that I use is applying different learning rates in different parts of the model: the learning rate in the fully connected part is still lr, last two blocks of ResNeXt are trained with lr/3, and first layers are trained with lr/10. Since low-level detectors do not vary much from one image data set to another, the first layers do not require substantial retraining compared to the parts of the model working with high level features. Another trick is learning rate annealing. Periodic learning rate increase followed by slow decrease drives the system out of steep minima (when lr is high) towards broader ones (which are explored when lr decreases) that enhances the ability of the model to generalize and reduces overfitting. The length of the cycles gradually increases during training. Usage of half precision doubles the maximum batch size that allows to compare more pairs in each batch.

# In[ ]:

# **Since the loss is calculated as an average of nonzero terms, as mentioned above, it's value is not relaiable and must be ignored.** Instead the values of T_acc and BH_acc metrics should be considered. 

# In[ ]:


def extract_embedding(model, path):
    tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO)
    tfms[0].tfms = [tfms[0].tfms[2], tfms[0].tfms[3]]
    tfms[1].tfms = [tfms[1].tfms[2], tfms[1].tfms[3]]
    ds = ImageData.get_ds(FilesDataset_single, (None, TRAIN), (None, TRAIN),
                          tfms, test=(None, path))
    md = ImageData(PATH, ds, 3 * bs, num_workers=nw, classes=None)
    model.eval()
    with torch.no_grad():
        preds = torch.zeros((len(md.test_dl.dataset), n_embedding))
        start = 0
        # for i, (x,y) in tqdm_notebook(enumerate(md.test_dl,start=0), total=len(md.test_dl)):
        for i, (x, y) in tqdm(enumerate(md.test_dl, start=0), total=len(md.test_dl)):
            #print(i)
            size = x.shape[0]
            m = model.module if isinstance(model, FP16) else model
            preds[start:start + size, :] = m.get_embedding(x.half())
            start += size
        return preds, [os.path.basename(name) for name in md.test_dl.dataset.fnames]

def cal_emb(model, dl):
    model.eval()
    with torch.no_grad():
        preds = torch.zeros((len(dl.dataset), n_embedding), dtype=torch.float16, device=device)
        start = 0
        #for i, (x, y) in tqdm(enumerate(dl, start=0), total=len(dl)):
        for i, (x, y) in enumerate(dl):
            #print(i)
            size = x.shape[0]
            m = model.module if isinstance(model, FP16) else model
            preds[start:start + size, :] = m.get_embedding(x.half())
            start += size
        return preds, [os.path.basename(name) for name in dl.dataset.fnames]


# ### **Validation**

# Find 16 nearest train neighbors in embedding space for each validation image. Since there can be several neighbors with the same label, instead of 5 I use 16 here. The following code will select 5 nearest neighbors with different labels. "new_whale" label can be assigned as a prediction at a distance dcut. In this case, if the number of neighbors at a distance shorter than dcut is less than 5, the image is considered to be different from others, and "new_whale" is assigned.

# In[ ]:

def get_nbs(model, x, y, n=16):
    d, idxs = [], []
    sz = x.shape[0]
    model.eval()
    with torch.no_grad():
        m = model.module if isinstance(model, FP16) else model
        m = m.module if isinstance(m, nn.DataParallel) else m
        for i in tqdm(range(sz)):
            preds = m.get_d(x[i], y)
            sorted, indices = torch.sort(preds)
            d.append(to_np(sorted[:n]))
            idxs.append(to_np(indices[:n]))
    return np.stack(d), np.stack(idxs)


def get_val_nbs1(model, emb_df, out='val.csv', dcut=None):
    emb_df = emb_df.copy()
    data = pd.read_csv(LABELS).set_index('Image')
    emb_df['emb'] = [[float(i) for i in s.split()] for s in emb_df['emb']]
    emb_df.set_index('files', inplace=True)
    train_df = data.join(emb_df)
    train_df = train_df.reset_index()
    # the split should be the same as one used for training
    trn_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    trn_preds = np.array(trn_df.emb.tolist())
    val_preds = np.array(val_df.emb.tolist())
    trn_df = trn_df.reset_index()
    val_df = val_df.reset_index()

    trn_preds = torch.from_numpy(trn_preds).half().cuda()
    val_preds = torch.from_numpy(val_preds).half().cuda()
    trn_d, trn_idxs = get_nbs(model, val_preds, trn_preds)

    s = []
    for l1 in trn_d.tolist():
        s.append(' '.join([str(l2) for l2 in l1]))
    val_df['d'] = s
    val_df['nbs'] = [' '.join(trn_df.loc[trn_idxs[index]].Id.tolist()) for index, row in val_df.iterrows()]
    val_df[['Image', 'Id', 'nbs', 'd']].to_csv(out, header=True, index=False)

    if dcut is not None:
        scores = []
        for idx in val_df.index:
            l0 = val_df.loc[idx].Id
            nbs = dict()
            for i in range(16):  # 16 neighbors
                nb = trn_idxs[idx, i]
                l, s = trn_df.loc[nb].Id, trn_d[idx, i]
                if s > dcut and 'new_whale' not in nbs: nbs['new_whale'] = dcut
                if l not in nbs: nbs[l] = s
                if len(nbs) >= 5: break
            nbs_sorted = list(nbs.items())
            score = 0.0
            for i in range(min(len(nbs), 5)):
                if nbs_sorted[i][0] == l0:
                    score = 1.0 / (i + 1.0)
                    break
            scores.append(score)
        print(np.array(scores).mean(), flush=True)
    return np.array(scores).mean()


def cal_val_dists(model, emb_df):
    emb_df = emb_df.copy()
    data = pd.read_csv(LABELS).set_index('Image')
    emb_df['emb'] = [[float(i) for i in s.split()] for s in emb_df['emb']]
    emb_df.set_index('files', inplace=True)
    train_df = data.join(emb_df)
    train_df = train_df.reset_index()
    # the split should be the same as one used for training
    trn_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    trn_preds = np.array(trn_df.emb.tolist())
    val_preds = np.array(val_df.emb.tolist())
    trn_df = trn_df.reset_index()
    val_df = val_df.reset_index()

    trn_preds = torch.from_numpy(trn_preds).half().cuda()
    val_preds = torch.from_numpy(val_preds).half().cuda()
    trn_d, trn_idxs = get_nbs(model, val_preds, trn_preds)

    return trn_df, val_df, trn_d, trn_idxs

def cal_val_score(trn_df, val_df, trn_d, trn_idxs, out='val.csv', dcut=None):
    s = []
    for l1 in trn_d.tolist():
        s.append(' '.join([str(l2) for l2 in l1]))
    val_df['d'] = s
    val_df['nbs'] = [' '.join(trn_df.loc[trn_idxs[index]].Id.tolist()) for index, row in val_df.iterrows()]
    val_df[['Image', 'Id', 'nbs', 'd']].to_csv(out, header=True, index=False)

    if dcut is not None:
        scores = []
        for idx in val_df.index:
            l0 = val_df.loc[idx].Id
            nbs = dict()
            for i in range(16):  # 16 neighbors
                nb = trn_idxs[idx, i]
                l, s = trn_df.loc[nb].Id, trn_d[idx, i]
                if s > dcut and 'new_whale' not in nbs: nbs['new_whale'] = dcut
                if l not in nbs: nbs[l] = s
                if len(nbs) >= 5: break
            nbs_sorted = list(nbs.items())
            score = 0.0
            for i in range(min(len(nbs), 5)):
                if nbs_sorted[i][0] == l0:
                    score = 1.0 / (i + 1.0)
                    break
            scores.append(score)
        #print(np.array(scores).mean(), flush=True)
    return np.array(scores).mean()

def find_threshold(model, emb_df, thresholds):
    trn_df, val_df, trn_d, trn_idxs = cal_val_dists(model, emb_df)

    best_threshold = 0
    max_score = 0
    for threshold in thresholds:
        score = cal_val_score(trn_df, val_df, trn_d, trn_idxs, out='val.csv', dcut=threshold)
        print(f'threshold: {threshold}: score: {score}')
        if score > max_score:
            max_score = score
            best_threshold = threshold
    print(f'best threshold {best_threshold}, best score {max_score}')
    return best_threshold, max_score



# In[ ]:

### **Submission**

# In[ ]:

def get_test_nbs(model, trn_emb, test_emb, out='test.csv', submission='submission.csv', dcut=None):
    print('generating submission')
    trn_emb = trn_emb.copy()
    data = pd.read_csv(LABELS).set_index('Image')
    trn_emb['emb'] = [[float(i) for i in s.split()] for s in trn_emb['emb']]
    trn_emb.set_index('files', inplace=True)
    train_df = data.join(trn_emb)
    train_df = train_df.reset_index()
    train_preds = np.array(train_df.emb.tolist())
    test_emb = test_emb.copy()
    test_emb['emb'] = [[float(i) for i in s.split()] for s in test_emb['emb']]
    test_emb['Image'] = test_emb['files']
    test_emb.set_index('files', inplace=True)
    test_df = test_emb.reset_index()
    test_preds = np.array(test_df.emb.tolist())
    train_preds = torch.from_numpy(train_preds).half().cuda()
    test_preds = torch.from_numpy(test_preds).half().cuda()
    test_d, test_idxs = get_nbs(model, test_preds, train_preds)

    s = []
    for l1 in test_d.tolist():
        s.append(' '.join([str(l2) for l2 in l1]))
    test_df['d'] = s
    test_df['nbs'] = [' '.join(train_df.loc[test_idxs[index]].Id.tolist()) for index, row in test_df.iterrows()]
    test_df[['Image', 'nbs', 'd']].to_csv(out, header=True, index=False)

    if dcut is not None:
        pred = []
        for idx, row in test_df.iterrows():
            nbs = dict()
            for i in range(0, 16):
                nb = test_idxs[idx, i]
                l, s = train_df.loc[nb].Id, test_d[idx, i]
                if s > dcut and 'new_whale' not in nbs: nbs['new_whale'] = dcut
                if l not in nbs: nbs[l] = s
                if len(nbs) >= 5: break
            nbs_sorted = list(nbs.items())
            p = ' '.join([lb[0] for lb in nbs_sorted])
            pred.append({'Image': row.files, 'Id': p})
        pd.DataFrame(pred).to_csv(submission, index=False)


def emb2file(model, path=TRAIN, emb_file='train_emb.csv'):
    emb, names = extract_embedding(model, path)
    df = pd.DataFrame({'files': names, 'emb': emb.tolist()})
    df.emb = df.emb.map(lambda emb: ' '.join(list([str(i) for i in emb])))
    df.to_csv(emb_file, header=True, index=False)

# In[ ]:


# ### **Hard negative example mining **

# The code below selects the most similar negative examples to images in the dataset. **They are really tough**, and as [Haider Alwasiti](https://www.kaggle.com/hwasiti) wrote in comments, I feel really sorry for the network to do such job. The following stage of training is performed on triplets with these hard negative example. Since distribution of these examples is different from one used at the previous stage of training, not only loss, but also metrics (T_acc, BH_acc) don't give a reliable estimation of the model performance, and only a way to check the model is validation based on the entire validation dataset with using the same metric as one in the competition.

# In[ ]:


# In[ ]:
# In[ ]:
