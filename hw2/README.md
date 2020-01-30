# Intro to pytorch

**Due:** Tuesday, 4 Feb at midnight

## Required tasks

Modify the `classifier.py` file so that:

1. Adjust the training procedure so that the test set is evaluated at the end of every epoch.

1. Implement three new models: the random feature model and the 1 hidden layer neural network model.
   You should add a command line argument `--model` which takes one of four options
   (`linear`, `factorized_linear`, `random_feature`, and `nn`)
   and another argument `--size` which takes takes an integer argument and controls the number of random features or the size of the hidden layer, depending on the model.
   This will require modifying the `define the model` and the `optimization` sections of the homework file.

1. Add a command line option to use the MNIST dataset instead of CIFAR10 for training and testing.
   (This will require changing code in both the `load dataset` and the `define the model` sections of code.)
   Torchvision has many other datasets as well (see https://pytorch.org/docs/stable/torchvision/datasets.html), and you can add these datasets too.

You should experiment with different values of `--alpha`, `--epochs`, `--batch_size`, `--model`, and `--size` to see how they effect your training time and the resulting accuracy of your models.
Try to find the combination that results in the best training accuracy.

## Recommended tasks

You are not required to complete the following tasks,
however they are good exercises to get you familiar with pytorch.

1. Currently, the print statement of the inner loop of the optimization prints the loss of a single batch of data.
   Because this is only a single batch of data, the loss value is highly noisy, and it is difficult to tell if the model is converging.
   The [exponential moving average](https://en.wikipedia.org/wiki/Moving_average) is a good way to smooth these values,
   and machine learning practitioners typically use this technique to smooth the training loss and measure convergence.
   Implement this technique in your `classifier.py` file.

1. Make the optimization use SGD with momentum.
   Add a command line flag that controls the strength of the momentum,
   and experiment to find a good momentum value.
   (Beta = 0.9 is often used.)

1. Add a "deep" neural network as one of the possible classifiers that has more than 1 hidden layer.
   Make the number of layers and the size of each layer a parameter on the command line.

## Submission

Upload your `classifier.py` file to sakai

