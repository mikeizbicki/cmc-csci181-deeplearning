# Name Generator

In this assignment, you will create a model that can generate new names that have the characteristics of different nationalities.

**Due:** Thursday, 9 April at midnight

**Learning Objective:**

1. understand the basics of generative neural networks
1. understand how machine learning researchers approach debugging code
1. effectively use tensorboard to monitor the training of models and select hyperparameters

## Tasks

The following tasks are required:

1. In the following youtube videos, I explain the theory behind generative recurrent neural network models and live code a pytorch implementation.
    You should follow along with the coding on your own computer so that you also have a working implementation.

    My video playlist:

    1. https://www.youtube.com/playlist?list=PLSNWQVdrBwob5e4UwLQ28ahtccaw0yiZ5

    Udacity also has a [great video on the beam search algorithm](https://www.youtube.com/watch?v=UXW6Cs82UKo) that you should watch.

1. Train two generative models, one using the `--conditional_model` flag and one without.
    In the lecture videos, I used a GRU network with the SGD optimizer.
    You must use both a different network and optimizer
    (for example use an LSTM and the Adam optimizer).
    This is so that you'll have to find different hyperparameters than the ones I used in the example to make you model work well.

    For each model, you should follow the training procedure we used in hw6:
    Train using the initial learning rate for 1e5 samples;
    then reduce the learning rate by a factor of 10 and repeat training;
    and finally reduce the learning rate again and repeat training.

    Your final models must have:
    1. `loss_nextchars` < 1.5 (for both models)
    1. `accuracy` >= 0.8 (only for the model without the `--conditional_model` flag)

1. Use tensorboard.dev to upload your tensorboard training runs to the cloud.
    You must create two separate tensorboard.dev webpages,
    one for each of the two sets of models trained above.
    Each of the tensorboard.dev pages should not have unrelated training runs appear in the plots.

    Here are the examples that I created in the videos:
    1. unconditional model: https://tensorboard.dev/experiment/AMAd6axxQEuP2P20PIDKpQ/#scalars&_smoothingWeight=0.99
    1. conditional model: https://tensorboard.dev/experiment/x99kAW5cQQ2NMgwU0lOdDQ/#scalars&_smoothingWeight=0.9
    <!--
    1. conditional model: https://tensorboard.dev/experiment/iW3xH64ZQ2yuLKkTpl8KuA/#scalars&_smoothingWeight=0.99
    -->

### Optional tasks

1. Modify the `CNNModel` class so that it also predicts the next character.

    In the video lectures, we only discussed how to modify the `RNNModel` class to predict the next character.
    The `CNNModel` class can also be used for predicting the next character in the same way.
    (And in fact, using `--model=cnn` currently results in a crash because of our changes to the training code.)

    *Bonus question:*
    Using a `CNNModel` to predict the next character is special case of using a *markov chain* to predict the next character of text.
    Why?

1. In the video lectures, I mention that you can use rejection sampling to sample from the conditional distributions when the model is an unconditional model.
    Implement this algorithm.

1. Implement beam search in your generation code.

## Submission

1. Submit the links to your tensorboard.dev pages on sakai.
