# Few-shot learning exploration

This repo is an exploration of some few-shot learning techniques as applied to the `Fashion Products` dataset available [here](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1). First, we will try to classify the products using a basic model, with some experiments designed to improve classification on rare classes:

## Part 1 - training a basic model

### Training on top20

To avoid dependencies, we decided not to use a 3rd party "training loop wrapper" library such as [fastai](https://github.com/fastai/fastai) or [ignite](https://github.com/pytorch/ignite) - our basic training loop logic is in the `fewshot.trainer` module: [trainer.py](./fewshot/trainer.py).

All data processing logic is in the `fewshot.data` module: [data.py](./fewshot/data.py). It relies on the instanciation of a `FashionData` class, specifying a `top20` flag which will determine if the datasets will contain the 20 most frequent classes, or the remaining rarest classes. This object will run all preprocessing, and instanciate training and test inheriting from pytorch `Datasets`.

All training was done with basic train-time augmentation (horizontal flipping and random cropping).

The final training runs were all done in [task1.ipynb](./notebooks/task1.ipynb), and show the training process. We outline below the main results from this notebook.

We first start by training the model on the most frequent classes. The training method and model selection procedure was as follows:

1. Train model for 20 epochs
2. Select two best models based on class-size weighed accuracy
3. Report model that had best top-1 average accuracy across classes

In hindsight, this procedure might have been improved by selecting directly on the desired accuracy metric (which would depend on the business application...) rather than going back on forth on weighed and unweighed accuracy.

We repeated this procedure across four losses:
- unweighed cross entropy
- class-weighed cross entropy
- unweighed focal loss (see [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002))
- class-weighed focal loss

The detailed results are in [task1.ipynb](./notebooks/task1.ipynb) - but here is a summary of average top-1 and top-5 accuracies across classes:

| Loss | Top-1 Accuracy | Top-5 Accuracy |
| ------------- | ------------- | --- |
| Unweighed cross entropy  |  84.4 | 94.1 |
| Class-weighed cross entropy | 84.8  | 94.7 |
| Unweighed Focal Loss | 84.7  | 94.1 |
| Class-weighed Focal Loss |  85.2 | 94.4 |

We therefore selected the class-weighed focal loss model as the best model. With more time I would have liked to check that this variation in result was actually statistically significant.

### Fine-tuning for the rare classes 

We then took the best model from the training above, and fine-tuned it on the rare classes using the same procedure as above. We compare the results of the entire procedure to results not using any fine-tuning (all with the weighed focal loss):

| Method | Top-1 Accuracy | Top-5 Accuracy | Weighed Top-1 Accuracy |
| --- | --- | ---| ---|
| With fine-tuning | 30.9 | 45.8 | 40.8 |
| Without fine-tuning | 17.4 | 39.0 | 27.6 |

### Discussion - and things I would have done with more time:

Beyond the extensions suggested, here are some things I would have liked to try:
- fix the training and model selectin procedure to rely on a single metric rather than go back and forth between two
- use more data augmentati:on. I almost added some color/saturation variations but then thought it could be problematic for some classes; for example Jeans would become a problem category if I changed their colors. I erred on the cautious side and didn't add it. Similarly affine transformations - some of these items, like watches or lipstics, are quite-geometry dependent so I didn't use any affine transformation.
- change the fine-tuning strategy to a twop-step strategy - first train only the FC layer, than train all layers together
- use a smarter LR scheduling policy, like one-cycle

## Part 2 - Few shot learning

The idea is to try out three few-shot papers on the fashion products dataset: [Prototypical Networks](https://arxiv.org/pdf/1703.05175.pdf), [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf), and [Model-Agnostic Meta-Learning](https://arxiv.org/pdf/1703.03400.pdf) (or MAML).

A series of [blog posts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77) with their [accompanying github repo](https://github.com/oscarknagg/few-shot) show the application on these methods to the two archetypal few-shot datasets, Omniglot and miniImageNet.


### Prototypical Networks

For now we've reimplemented the dataset as applied to the Fasion Products dataset and added an episode viz function in `fewshot.proto.sampler`. Corresponding exploration notebook is [here](./notebooks/proto-exploration.ipynb).

Our first iteration of the training loop, in the exploration notebook, gets to these results with a resnet18 architecture and a 100-dimensional embedding space:

|Loss | Top-1 Val. Accuracy |
| ------------- | ------------- |
| 2-shot, 20-way  |  72.9 |
| 1-shot, 20-way | 61.1  |
| 2-shot, 5-way  |  88.4 |
| 1-shot, 5-way | 84.4  |

Comparing these results to those in the paper, it seems that the 'difficulty' of this fashion dataset is somewhere between Omniglot and miniImageNet. That said, they're extremely good results (maybe too good - check for leakage?).

### Prototypical Networks - ideas and extensions

To improve our results on this task, I'm thinking of a couple of new approaches:
1. **Better metric for distances on the embedding manifold:** The embedding space is definitely not flat - so using plain old L2 distance seems like it wouldn't be the most appropriate. Inspired by Natural Gradient Descent, whose idea is to normalize gradients by the curvature of the space as estimated by the Fisher (see e.g. [Martens 2014](https://arxiv.org/abs/1412.1193)), we would like to compute the prototypes using a better metric than L2. That said, the empirical methods to compute the Fisher seem to be quite bad [Limitations of the Empirical Fisher Approximation](https://arxiv.org/abs/1905.12558).

The idea would then be to resort to a heuristic normalization (maybe a simple diagonal pre-conditioning like in RMSProp) to compute the distances. 

2. **Data augmentation strategies** It would be interesting to see the impact of traditional data augmentation strategies. In addition, two interesting areas to explore would be:
- Using adversarial examples as augmentation, as in the semi-supervised approach proposed in [Virtual Adversarial Training](https://arxiv.org/pdf/1704.03976.pdf) - this technique had the best results in [this evaluation of deep semi-supervised learning](https://arxiv.org/abs/1804.09170)
- Using [mixup](https://arxiv.org/abs/1710.09412) - seems like it would be particularly important for the embedding space to be linearly separable, and using mixup data augmentation would maybe help with that?

That said, I'll probably look at MAML before implementing these two ideas.

## MAML






