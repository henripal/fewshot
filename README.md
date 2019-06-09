# Few-shot learning exploration

This repo is an exploration of some few-shot learning techniques as applied to the `Fashion Products` dataset available [here](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset/version/1). First, we will try to classify the products using a basic model, with some experiments designed to improve classification on rare classes:

## Part 1 - training a basic model

To avoid dependencies, we decided not to use a 3rd party "training loop wrapper" library such as [fastai](https://github.com/fastai/fastai) or [ignite](https://github.com/pytorch/ignite) - our basic training loop logic is in the `fewshot.trainer` module: [trainer.py](./fewshot/trainer.py).

All data processing logic is in the `fewshot.data` module: [data.py](./fewshot/data.py). It relies on the instanciation of a `FashionData` class, specifying a `top20` flag which will determine if the datasets will contain the 20 most frequent classes, or the remaining rarest classes. This object will run all preprocessing, and instanciate training and test inheriting from pytorch `Datasets`.

The final training runs were all done in [task1.ipynb](./notebooks/task1.ipynb), and show the training process. We outline below the main results from this notebook.

We first start by training the model on the most frequent classes



