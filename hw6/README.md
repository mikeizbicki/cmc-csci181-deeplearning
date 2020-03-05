# Name Classifier

In this assignment, you will create a model that can predict the nationality of surnames.

**Due:** Thursday, 12 March at midnight

**Learning Objective:**

1. gain familiarity with character-level text models (RNN / CNN)
1. effectively use tensorboard to monitor the training of models and select hyperparameters

## Tasks

Complete the following required tasks.

1. **Download the starter code and data:**
   On the command line, run:
   ```
   $ wget https://github.com/mikeizbicki/cmc-csci181/blob/master/hw6/names.py
   $ wget https://github.com/mikeizbicki/cmc-csci181/blob/master/hw6/names.tar.gz
   $ tar -xf names.tar.gz
   ```
   You should always manually inspect the data values before performing any coding or model training.
   In this case, get a list of the data files by running
   ```
   $ ls names
   ```
   Notice that there is a file for several different nationalities.
   Inspect the values of some of these files.
   ```
   $ head names/English.txt
   $ head names/Spanish.txt
   $ head names/Korean.txt
   ```
   Notice that the names have been romanized for all languages,
   but the names are not entirely ASCII.
   We will revisit this fact later.

   Also notice that each line contains a unique name.
   To get the total number of lines in each file (and therefore the total number of examples for each class), run
   ```
   $ wc -l names/*
   ```
   Observe based on this output that the training data is not equally balanced between classes.

1. **Different learning rates:**
   At the command prompt, execute the following line:
   ```
   $ python3 names.py --train --learning_rate=1e-1
   ```
   In a separate command prompt, launch tensorboard with the line:
   ```
   $ tensorboard --logdir=log
   ```
   You should observe that the loss function is diverging.
   Experiment with different loss functions to find the optimal value.

   **NOTE:**
   In order to easily interpret the tensorboard plots, you may have to increasing the smoothing paramaeter very close to 1.
   I used a value of 0.99.

   **Question 1:**
   What is the optimal learning rate you found?

1. **Gradient clipping:**
   Tensorboard is recording three values: the training accuracy, the training loss, and the norm of the gradient.
   Notice that as training progresses, the norm of the gradient increases.
   This is called the *exploding gradient problem*.

   The standard solution to the exploding gradient problem is *gradient clipping*.
   In gradient clipping, we first measure the L2 norm of the gradient;
   then, if it is larger than some threshold value, we shrink the gradient so that it points in the same direction but has norm equal to the threshold.

   To add support for gradient clipping to your code,
   paste the following lines just before the call to `optimizer.step`.
   ```
    if args.gradient_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
   ```

   Rerun the model training code, but now with the `--gradient_clipping` flag to enable gradient clipping.
   Once again, experiment to find the optimal value for the learning rate.

   **Question 2:**
   What is the new optimal learning rate you found with gradient clipping enabled?

   **Question 3:**
   Which set of hyperparameters is converging faster?

   At this point, hopefully this XKCD comic is starting to make sense:
   <p align=center>
   <img src=https://imgs.xkcd.com/comics/machine_learning_2x.png width=400px>
   </p>

1. **Optimization method:**
   [Adam](https://arxiv.org/abs/1412.6980) is a popular alternative to SGD for optimizing models.

   To add support for the Adam optimizer to the code,
   paste the following lines below the code for the SGD optimizer.
   ```
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
                )
   ```

   Use the `--optimizer=adam` flag to train a model using Adam instead of SGD.
   Like SGD, Adam takes a learning rate hyperparameter,
   and you should experiment with different values to find the optimal value.

   **Question 4:**
   What is the optimal learning rate for Adam?

   **Question 5:**
   Which set of hyperparameters is converging faster?

1. **Different types of RNNs:**
   There are three different types of RNNs is common use.
   So far, your model has been using "vanilla" RNNs,
   which is what we discussed in class.
   Two other types are *gated recurrent units* (GRUs) and *long short term memories* (LSTMs).
   GRUs and LSTMs have more complicated activation functions that try to better capture long-term dependencies within the input text.
   Visit [this webpage](https://medium.com/@saurabh.rathor092/simple-rnn-vs-gru-vs-lstm-difference-lies-in-more-flexible-control-5f33e07b1e57) to see a picture representation of each of the RNN units.

   Understanding in detail the differences between the types of RNNs is not important.
   What is important is that the inputs and outputs of vanilla RNNs, GRUs, and LSTMs are all the same.
   This means that you can easily switch between the type of recurrent network by simply calling the appropriate torch library function.
   (The functions are `torch.nn.RNN`, `torch.nn.GRU`, and `torch.nn.LSTM`.)

   Adjust the `Model` class so that it uses either RNNs, GRUs, or LSTMs depending on the value of the `--model` input parameter.

   **Question 6:**
   Once again, experiment with different combinations of hyperparameters using these new layers.
   What gives the best results?

1. **RNN size:**
   There are two hyperparameters that control the "size" of an RNN.
   The advantages of larger sizes are: better training accuracy and improved ability to capture longterm dependencies within text.
   The disadvantages are: longer training time and worse generalization error.

   Adjust the `Model` class so that the calls to `torch.nn.RNN`,`torch.nn.GRU`, and `torch.nn.LSTM` use the `--hidden_layer_size` and `--num_layers` command line flags to determine the size of the hidden layers and number (respectively).

   **Question 7:**
   Experiment with different model sizes to find a good balance between speed and accuracy.

1. **Change batch size:**
   Currently, the training procedure uses a fixed batch size with only a single training example.
   There are two key functions for generating batch tensors from strings.

   The first is `unicode_to_ascii`, which converts a Unicode string into a Latin-only alphabet representation.
   Run this command on the strings `Izbicki`, `Ízbìçkï` and `이즈비키` to see how they get processed and the limitations of our current system.

   The second is `str_to_tensor`, which converts an input string into a 3rd order tensor.
   Notice that the first dimension is for the length of the string, and the second dimension is for the batch size.
   This is a standard pytorch convention.

   Modify the `str_to_tensor` function so that it takes a list of b strings as input instead of only a single string.
   The input strings are unlikely to be all of the same length,
   but the output tensor must have the same length for each string.
   To solve this problem, the first dimension of the tensor will have the largest size of all of the input strings;
   then, the remaining input strings will have their slices padded with all zeros to fill the space.
   To help the model understand when it reaches the end of a name, the special `$` character is used to symbolize the end of a name, and this should be inserted at the end of each string, before the zero padding.

   Next, you will need to modify the data sampling step to sample `args.batch_size` data points on each iteration.
   This is the part of the code after the comment `# get random training example`.

   **Question 8:**
   Experiment with a larger batch size.
   This should make training your model a bit faster because the matrix multiplications in your CPU will have better cache coherency.
   (A single step of batch size 10 should take only about 5x longer than a step of batch size 1.)
   A larger batch size will also reduce the variance of each step of SGD/Adam, and so a larger step size can be used.
   As a rule of thumb, increasing the batch size by a factor of `a` will let you increase the learning rate by a factor of `a` as well.

   With a batch size of 16, what is the new optimal learning rate?

1. **Add CNN support:**
   CNNs typically have a linear layer on top of them,
   and this linear layer requires that all inputs have a fixed length.

   1. Modify the `str_to_tensor` function so that if the `--input_length` parameter is specified,
      then the tensor is padded/truncated so that the first dimension has size `args.input_length`.

   1. Modify the `Model` class so that if the `--model=cnn` parameter is specified,
      then a cnn is used (the `torch.nn.Conv1d` function).
      Your implementation should use a width 3 filter.
      `--hidden_layer_size` as the number of channels,
      and your should have `--num_layers` cnn layers.

   **Question 9:**
   Experiment with different hyperparameters to find the best combination for the CNN model.
   How does the CNN model compare to the RNN models?

1. **Longrun model training:**
   Once you have a set of model hyperparameters that you like,
   then increase `--samples` to 100000 to train a more accurate model.
   Then, use the `--warm_start` parameter to reload this model,
   and train for another 100000 samples (but this time with a learning rate lowered by a factor of 10).
   Repeat this procedure one more time.

   The whole procedure should take 10-30 minutes depending on the speed of your computer and the complexity of your model.
   This would be a good point to have an office chair jousting dual
   <p align=center>
   <img src=img/xkcd-training.png>
   </p>
   (Comic modified from https://xkcd.com/303/)

1. **Inference:**
   You can use the `--infer` parameter combined with `--warm_start` to use the model for inference (sometimes called model *deployment*).
   In this mode, `names.py` passes each line in stdin to the model and outputs the class predictions.

   You should modify the inference code so that instead of outputting a single prediction,
   it outputs the model's top 3 predictions along with the probability associated with each prediction.

   To get more than the top 1 prediction, you will have to change how the `topk` function is called.

   To convert the `output` tensor into probabilities, you will have to apply the `torch.nn.Softmax` function.

## Submission

Upload your code to sakai.
Hand in a hard copy of your completed answers.
