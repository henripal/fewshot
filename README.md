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

There is maybe a bug in the author's implementation of the data loading [here](https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/few_shot/proto.py#L50) - `queries` should be `embedding[q_queries * k_way:]` not `embedding[n_shot * k_way]`. See the corresponding data generation step [here](https://github.com/oscarknagg/few-shot/blob/672de83a853cc2d5e9fe304dc100b4a735c10c15/few_shot/core.py#L81).

To avoid chasing bugs while not wasting time reimplementing everything from scratch, I'll be doing a minimal / loose reimplementation strongly based on @oscarknagg's repo. 

### Prototypical Networks

For now we've reimplemented the dataset as applied to the Fasion Products dataset (without the bug :)) and added an episode viz function in `fewshot.proto.sampler`. Corresponding exploration notebook is [here](./notebooks/proto-exploration.ipynb).









