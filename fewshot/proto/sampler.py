## Addapted from https://github.com/oscarknagg/few-shot/blob/master/few_shot/core.py
## Modified to work with our dataset

from typing import List, Iterable, Callable, Tuple
import os

import torch.utils.data
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..data import FashionDataset
from ..data import tensor2numpy

class NShotFashionDataset(FashionDataset):
    """
    child class of FashionDataset, adding some features necessary for sampling
    """
    def __init__(self, csv_file, images_root, classlist=None, transform=None):
        self.label_data = pd.read_csv(csv_file)
        super(NShotFashionDataset, self).__init__(self.label_data, images_root, transform)
        self.imagelist = [imagefile.split('.')[0] for imagefile in os.listdir(images_root) if imagefile.split('.')[0] != '']
        self.filter_existing_images()
        self.classlist = classlist
        if classlist: self.filter_classlist()
        self.make_label_utils()

    def filter_existing_images(self):
        """
        truncates the dataframe to images that are found in path
        """
        self.label_data = self.label_data[self.label_data.id.isin(self.imagelist)]
        if len(self.label_data) == 0: raise FileNotFoundError('Image directory is empty or does not match csv file')

    def filter_classlist(self):
        self.label_data = self.label_data[self.label_data.articleType.isin(self.classlist)]

    def make_label_utils(self):
        self.idx2name = self.label_data.articleType.value_counts().index.values
        self.name2idx = {name: idx for idx, name in enumerate(self.idx2name)}
        self.n_classes = len(self.idx2name)



class NShotTaskSampler(torch.utils.data.Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 ):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.
        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.
        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
        
        HP comments:
        refactored this to use tensors directly and to work with our FashionDataset class
        I'm not a fan of putting everything in one tensor but running with it
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset

        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q


        self.i_task = 0


    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []


            # Get random classes
            counts = self.dataset.label_data.articleType.value_counts()
            potential_classes = counts[counts > (self.n + self.q)].index.values

            episode_classes = np.random.choice(
                potential_classes,
                size=self.k,
                replace=False
                )

            support_k = {k: None for k in episode_classes}

            for k in episode_classes:
                # which elements of the dataset are in k?
                indices_k = self.dataset.label_data[
                    self.dataset.label_data.articleType == k
                    ].index.values
                support = np.random.choice(indices_k,
                    size=self.n,
                    replace=False) 
                support_k[k] = support

                for idx in support:
                    batch.append(idx)

            for k in episode_classes:
                indices_k = self.dataset.label_data[
                    self.dataset.label_data.articleType == k
                    ].index.values
                support = support_k[k]
                indices_k = [idx for idx in indices_k if idx not in support]
                query = np.random.choice(indices_k,
                    size=self.q,
                    replace=False)
                for idx in query:
                    batch.append(idx)

            yield batch


def plot_nshot_episode(episode, n, k, q, scale):
    fig, axs = plt.subplots(k, n+q, sharex='col', sharey='row',
                        gridspec_kw={'wspace': 0},
                        subplot_kw={'xticks': [], 'yticks': []},
                        figsize=(scale*(n+q), scale*k))
    idx = 0 
    for i in range(k):
        for j in range(n): 
            axs[i, j].imshow(tensor2numpy(episode[0][idx]), aspect='auto')
            idx += 1
            

            
    for i in range(k):
        for j in range(q):
            axs[i, n + j].imshow(tensor2numpy(episode[0][idx]), aspect='auto')
            idx += 1

    for ax, row in zip(axs[:,0], episode[1][::n]):
        ax.set_ylabel(row)
        
    col_titles = ['example {}'.format(n_ex) for n_ex in range(n)] + ['query {}'.format(q_ex) for q_ex in range(q)]
    for ax, col in zip(axs[0,:], col_titles):
        ax.set_title(col)

    fig.tight_layout()
    
