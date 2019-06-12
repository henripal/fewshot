import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import datetime
import csv
import os

def run_training(model, n_epochs, n, k, q, train_dl, test_dl, optimizer, gpu_id=0, root='./'):
    """
    basic training loop. not super customizable but has checkpointing and saves training traces
    will create a folder and save models and training trace in that folder
    """
    loss_func = nn.CrossEntropyLoss()

    torch.cuda.set_device(gpu_id)
    model.cuda();
    folder = root + 'gpu_' + str(gpu_id) + '_' + datetime.datetime.now().strftime("%b%d_%H%M") + '/'
    if not os.path.exists(folder): os.mkdir(folder)

    for epoch in range(n_epochs):
        print('------------- epoch: ', epoch)
        
        # training
        running_losses = []
        model.train()
        for images, _ in train_dl:
            images = images.cuda()
            labels = torch.arange(0, k, 1/q).long().cuda()

            optimizer.zero_grad()
            embeddings = model(images)
            n_embeds = embeddings[:n*k]
            q_embeds = embeddings[n*k:]

            prototypes = compute_centroids(n_embeds, n, k)
            distances = pairwise_distances(prototypes, q_embeds, matching_fn="l2").t()

            loss = loss_func(-distances, labels)
            running_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        train_loss = '{:.3f}'.format(np.mean(running_losses))
        print('Training loss:', train_loss)
            
        # validation
        model.eval()
        running_val_losses = []
        running_accuracies = []
        for images, _ in test_dl:
            with torch.no_grad():
                images = images.cuda()
                labels = torch.arange(0, k, 1/q).long().cuda()
                embeddings = model(images)
                n_embeds = embeddings[:n*k]
                q_embeds = embeddings[n*k:]

                prototypes = compute_centroids(n_embeds, n, k)
                distances = pairwise_distances(prototypes, q_embeds, matching_fn="l2").t()

                loss = loss_func(-distances, labels)
                running_val_losses.append(loss.item())

                _, predicted = torch.max(-distances, 1)
                running_accuracies.append((predicted == labels).float().mean().item())
            
        val_loss = '{:.3f}'.format(np.mean(running_val_losses))
        print('Validation loss:', val_loss)
        val_acc = '{:.3f}'.format(np.mean(running_accuracies))
        print('Validation accuracy:', val_acc)

        # checkpointing

        with open(folder + 'trace.csv', 'a+', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0: writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])
            writer.writerow([str(epoch), train_loss, val_loss, val_acc])

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, folder + 'model_epoch_' + str(epoch) + '.pth')

def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


def compute_centroids(embeddings, n, k):
    l, embed_sz = embeddings.shape
    assert l == k * n
    examples = embeddings[:n*k].reshape(k, -1, embed_sz)
    # examples has shape k, n, embed_sz, and we average across the n dimension
    return examples.mean(1)