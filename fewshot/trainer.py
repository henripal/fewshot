import torch
import numpy as np
import datetime
import csv
import os


def run_training(model, n_epochs, data, optimizer, loss_func, gpu_id=0, root='./', bs=64):
    """
    basic training loop. not super customizable but has checkpointing and saves training traces
    will create a folder and save models and training trace in that folder
    """
    train_dl = torch.utils.data.DataLoader(data.train_ds,
                                 batch_size=bs, shuffle=True,
                                 num_workers=4)
    test_dl = torch.utils.data.DataLoader(data.test_ds,
                                    batch_size=bs*2, shuffle=False,
                                    num_workers=4)


    torch.cuda.set_device(gpu_id)
    model.cuda();
    folder = root + 'gpu_' + str(gpu_id) + '_' + datetime.datetime.now().strftime("%b%d_%H%M") + '/'
    if not os.path.exists(folder): os.mkdir(folder)

    for epoch in range(n_epochs):
        print('------------- epoch: ', epoch)
        
        # training
        running_losses = []
        model.train()
        for images, labels in train_dl:
            images = images.cuda()
            labels = torch.tensor([data.name2idx[label] for label in labels]).cuda()
            optimizer.zero_grad()
            out = model(images)

            loss = loss_func(out, labels)
            running_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        train_loss = '{:.3f}'.format(np.mean(running_losses))
        print('Training loss:', train_loss)
            
        # validation
        model.eval()
        running_val_losses = []
        running_accuracies = []
        for images, labels in test_dl:
            with torch.no_grad():
                images = images.cuda()
                labels = torch.tensor([data.name2idx[label] for label in labels]).cuda()
                out = model(images)
                loss = loss_func(out, labels)
                running_val_losses.append(loss.item())
                _, predicted = torch.max(out.data, 1)
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

def evaluate(model, data, gpu_id, n_classes, bs=64):
    """
    runs top-1 and top-5 accuracy on test set for the model for each class
    """
    test_dl = torch.utils.data.DataLoader(data.test_ds,
                                    batch_size=bs*2, shuffle=False,
                                    num_workers=4)
    model.eval()

    occurences = torch.zeros(n_classes).cuda()
    correct1_all = torch.zeros(n_classes).cuda()
    correct5_all = torch.zeros(n_classes).cuda()

    for images, labels in test_dl:
        with torch.no_grad():
            images = images.cuda()
            labels = torch.tensor([data.name2idx[label] for label in labels]).cuda()
            out = model(images)
            _, pred = out.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            correct1 = correct[:1].view(-1).float()
            correct5 = correct[:5].float().sum(0)

            correct1_all[labels] += correct1
            correct5_all[labels] += correct5
            occurences[labels] += 1

    return correct1_all/occurences, correct5_all/occurences, occurences

def pretty_print_eval(c1, c5, occ,  data):
    """
    prints evaluation data in a table
    takes the averge of the accuracies for classes for which there is at least 1 example in the data.
    """
    c1, c5, occ  = c1.cpu().numpy(), c5.cpu().numpy(), occ.cpu().numpy()
    mask = occ > 0
    print('{:23}{:>10}{:>10}'.format('Category','Top-1','Top-5'))
    print('-'*43)
    for i, cname in enumerate(data.idx2name):
        if mask[i]:
            print('{:23}{:10.1f}{:10.1f}'.format(cname,c1[i]*100,c5[i]*100))
    print('='*43)
    print('{:23}{:10.1f}{:10.1f}'.format('Average',np.nanmean(c1[mask])*100,np.nanmean(c5[mask])*100))

        

