import csv
import os
import PIL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision



def fix_quotes(in_csv, out_csv):
    """
    fixes the problem with commas in last column of csv by quoting last column of csv
    in_csv: string path to the raw csv file
    out_csv: string path to the desired output file.
        Will be created if not already existing. If already existing, file will be overwritten
    """
    with open(in_csv) as f, open(out_csv, 'w+') as o:
        reader, writer = csv.reader(f), csv.writer(o)
        for line in reader:
            newline = line[:9] + [','.join(line[9:])]
            writer.writerow(newline)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

def tensor2img(tensor):
    """
    converts tensor and displays image
    """
    rescaled = (tensor - tensor.min())/(tensor.max() - tensor.min())
    plt.imshow(rescaled.numpy().swapaxes(0, 2).swapaxes(0, 1))

class FashionData:
    def __init__(self, csv_file, images_root, train_transform=None, test_transform=None, top20=False):
        self.images_root = images_root
        self.top20 = top20
        self.csv_file = csv_file
        
        self.imagelist = [imagefile.split('.')[0] for imagefile in os.listdir(images_root) if imagefile.split('.')[0] != '']
        self.label_data = pd.read_csv(csv_file)
        self.filter_existing_images()
        self.filter_top20(top20)
        self.make_label_utils()
        self.calc_weights()
        
        
        self.train_ds = FashionDataset(self.label_data[self.label_data.year % 2 == 0].reset_index(drop=True),
                                      images_root,
                                      train_transform)
        self.test_ds = FashionDataset(self.label_data[self.label_data.year % 2 == 1].reset_index(drop=True),
                                     images_root,
                                     test_transform)
        
    def make_label_utils(self):
        self.idx2name = self.label_data.articleType.value_counts().index.values
        self.name2idx = {name: idx for idx, name in enumerate(self.idx2name)}
        self.n_classes = len(self.idx2name)

    def calc_weights(self):
        """
        calculate inverse frequency of classes
        """
        train_ld = self.label_data[self.label_data.year % 2 == 0]
        self.weights = np.zeros(self.n_classes)
        for label, count in train_ld.articleType.value_counts().iteritems():
            self.weights[self.name2idx[label]] = 1/count
            
    def filter_existing_images(self):
        """
        truncates the dataframe to images that are found in path
        """
        self.label_data = self.label_data[self.label_data.id.isin(self.imagelist)]
        if len(self.label_data) == 0: raise FileNotFoundError('Image directory is empty or does not match csv file')
    
    def filter_top20(self, top20):
        sorted_labels = self.label_data.articleType.value_counts().index.values
        if top20:
            self.label_data = self.label_data[self.label_data.articleType.isin(sorted_labels[:20])]
        else:
            self.label_data = self.label_data[self.label_data.articleType.isin(sorted_labels[20:])]




class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, label_data, images_root, transform=None):
        """
        csv_file: path to csv file containing the labels (styles.csv)
        images_root: path to the images
        transform: list of transforms we want to apply
        train: whether we want to get the train images or the test images
        top20: whether we want to limit the images to the top20 most frequent classes
        """
        self.images_root = images_root
        self.transform = transform
        self.label_data = label_data
            
    def __len__(self):
        return len(self.label_data)
    
    def __getitem__(self, idx):
        img_id, label = self.label_data.loc[idx, ['id', 'articleType']].values
        filename = os.path.join(self.images_root, str(img_id) + '.jpg')
        
        img = pil_loader(filename)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label