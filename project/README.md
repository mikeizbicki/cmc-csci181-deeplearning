# Project: analyzing news articles about the coronavirus

**Overview:**
This is the final project for [CMC's CS181: Deep Learning](https://github.com/mikeizbicki/cmc-csci181) course.
The project will guide you through the process of using state of the art deep learning techniques to analyze the news coverage of the coronavirus.
The dataset you will analyze contains 2 million news articles written in 20 languages and published in 50,000 venues around the world.
Despite this large dataset size,
the project has been designed to be completed on only modest hardware,
and specifically does not require access to a GPU.

**Scientific goals:**
We will try to answer the following questions:

1. What is the bias of different news sources? (geographic, topical, ideological, etc.)

1. How has coverage of the coronavirus changed over time?

1. Can we detect/generate "fake news" stories about the coronavirus?
    Wikipedia has a [big list of fake news stories related to coronavirus](https://en.wikipedia.org/wiki/Misinformation_related_to_the_2019%E2%80%9320_coronavirus_pandemic).

<!--
1. Given a particular meme can we track its spread through the media?
    1. **Geographic:** 
        The [LATimes](https://latimes.com) will publish different articles than the [Miami Herold](https://www.miamiherald.com), and both of these will be different than the [Choson Ilbo](http://english.chosun.com/) (a Korean newspaper).
        In order to understand the news about a given region,
        we first need to be able to find all the newspapers that report about a given region.

    1. **Topical:**

    1. **Ideological:**
        The Huffington Post is typically pro-imigration, and has run stories about [doctors detained by ICE complaining about conditions in detention centers](https://www.huffpost.com/entry/coronavirus-fear-in-immigrant-detention_n_5e8dd8b0c5b6e1d10a6cfa87),
        and the pro-isolationist InfoWars has run stories about [migrant labor causing the spread of the coronavirus](https://www.infowars.com/migrant-labor-was-to-blame-for-coronavirus-spread-in-both-iran-italy/).
-->

<!--
    For example, there have been thouands of news stories about facemasks and whether they help prevent the spread of coronavirus.
    At different times
-->

**Learning objectives:**

1. Have a cool project in your portfolio to talk about in job interviews

1. Understand the following deep learning concepts
    1. explainability
    1. attention (compared with RNNs and CNNs)
    1. transfer learning / fine tuning
    1. embeddings

1. Apply deep learning techniques to a real world dataset
    1. understand data cleaning techniques
    1. understand the importance of Unicode in both English and foreign language text
    1. learn how to use "natural supervision" to generate labels for unlabeled data
    1. learn how to approach a problem that no one knows the answers to

1. Understand the research process

**Related projects:**

There's been lots of machine learning research applied to the coronavirus ([see here for a really big list](https://towardsdatascience.com/machine-learning-methods-to-aid-in-coronavirus-response-70df8bfc7861)).
The closest related research to this project is:

1. Kaggle is hosting [a competition to analyze academic articles about coronavirus](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).
    The dataset includes 47,000 scientific articles,
    which is too many for doctors to read,
    and the goal is to extract the most relevant information about these articles.
1. Stanford hosted [a virtual conference on COVID-19 and AI](https://hai.stanford.edu/events/covid-19-and-ai-virtual-conference/agenda), the most relavent presentation was by [Renée DiResta](https://hai.stanford.edu/people/ren-e-diresta) on [Misinformation & Disinformation in state media about COVID-19](https://www.youtube.com/watch?v=z4105Exe23Q&t=1hr58m10s)

## Part 0: The data

You can download version 0 of the dataset at:

1. training set: https://izbicki.me/public/cs/cs181/coronavirus-headlines-train.jsonl.gz
1. test set: https://izbicki.me/public/cs/cs181/coronavirus-headlines-test.jsonl.gz

You should download the training set and place it in a directory called `coronavirus_headlines` with the following commands:
```
$ mkdir coronavirus-headlines
$ cd coronavirus-headlines
$ wget https://izbicki.me/public/cs/cs181/coronavirus-headlines-train.jsonl.gz
```

The dataset is stored in the [JSON Lines](http://jsonlines.org/) format,
where every line represents a single news articles and has the following keys:

| key           | semantics |
| ------------- | --------- |
| `url`         | The url of the article |
| `hostname`    | The hostname field of the url (e.g. `www.huffpost.com` or `www.breitbart.com`) |
| `title`       | The title of the article as extracted by [newspaper3k](https://newspaper.readthedocs.io/en/latest/).  This is typically the `<h1>` tag or the `<title>` tag of the webpage.  No preprocessing has been done on the titles, and so many titles contain weird unicode values that need to be normalized. | 
| `pub_time`    | The date of publication as extracted by [newspaper3k](https://newspaper.readthedocs.io/en/latest/).  These dates should be taken with a heavy grain of salt.  Many of these dates are clearly wrong, for example there are dates in the 2030s and dates from thousands of years in the past.  Furthermore, some domains like `sputniknews.com` use the European date convention of YYYY-MM-DD, but these are interpreted by newspaper3k as YYYY-DD-MM dates, and so have their day and month flipped.  My guess is that for dates that might be relevant to the coronavirus (roughly 2019-11-01 to 2020-04-01), somewhere between 50%-80% of the dates are correct. |
| `lang`        | The language ISO-2 code determined by applying [langid.py](https://github.com/saffsd/langid.py) to the body of the article.  For popular languages like English and Chinese, I think these labels are fairly accurate.  But for other languages I'm less confident.  For example, there are many articles about coronavirus labeled as Latin, and my suspicion is that most of these articles are actually written in another romance language like Spanish or Italian. |

This isn't every article ever written about the coronavirus,
but it's a large fraction of them.
The list has been filtered to include only English language articles,
and articles whose title contains one of the strings `coronavirus`, `corona virus`, `covid` or `ncov`.

## Part 1: explainable machine learning

**due date:** Thursday, 23 April

### What's already been done 
Our first goal when analyzing this dataset is to predict the hostname that published an article given just the title.
We will see that this gives us a simple (but crude) way to measure how similar two different hostnames are.

I trained this model with the following command:
```
$ python3 names.py                                                      \
    --train                                                             \
    --data=coronavirus-headlines/coronavirus-headlines-train.jsonl.gz   \
    --data_format=headlines                                             \
    --model=gru                                                         \
    --hidden_layer_size=512                                             \
    --num_layers=8                                                      \
    --resnet                                                            \
    --dropout=0.2                                                       \
    --optimizer=adam                                                    \
    --learning_rate=1e-3                                                \
    --gradient_clipping                                                 \
    --batch_size=128 
```
You do not have to run this command yourself, as it will take a long time.
(I let it run for 2 days on my GPU system.)
I am providing the command just so that you can see the particular hyperparameters used to train the model.

You should download the pretrained model by running the commands
```
$ wget https://izbicki.me/public/cs/cs181/gru_512x8.tar.gz
$ mkdir models
$ mv gru_512x8.tar.gz models
$ tar -xzf models/gru_512x8.tar.gz
``` 

The last step you need to do before running the code is create a directory for the output explanations to be saved into:
```
$ mkdir explain_outputs
```

We can now run inference on our model with the command
```
$ python3 names.py --infer --warm_start=models/gru_512x8
```
which will accept text from stdin and run the inference algorithm on it.

Examples:

1. Geographic similarity:
    The Australian website `www.news.com.au` ran the story titled
    ```
    Sick Qantas passenger at Melbourne Airport sparks coronavirus fears
    ```
    We can run our inference algorithm on this title using the command
    ```
    $ python3 names.py --infer --warm_start=models/gru_512x8 <<< "Sick Qantas passenger at Melbourne Airport sparks coronavirus fears"
    ```
    The top 5 predictions are:
    ```
       0 www.news.com.au (0.24)
       1 www.abc.net.au (0.04)
       2 www.dailymail.co.uk (0.04)
       3 au.news.yahoo.com (0.03)
       4 news.yahoo.com (0.03)
    ```
    Most of these are other Australian newspapers.

1. Topical similarity:
    The website `virological.org` is a discussion forum where doctors and biologists post their analysis of different viruses.
    Understandably, they have been recently been posting detailed analyses of the coronavirus,
    and one such post was titled
    ```
    nCoV's relationship to bat coronaviruses & recombination signals (no snakes) - no evidence the 2019-nCoV lineage is recombinant
    ```
    We can run our inference algorithm using the command
    ```
    $ python3 names.py --infer --warm_start=models/gru_512x8 <<< "nCoV's relationship to bat coronaviruses & recombination signals (no snakes) - no evidence the 2019-nCoV lineage is recombinant"
    ```
    The top 5 predictions are:
    ```
       0 virological.org (0.46)
       1 www.businessinsider.com (0.04)
       2 www.insider.com (0.02)
       3 contagiontracker.com (0.02)
       4 www.en24.news (0.01)
    ```
    This suggests that these websites all publish relatively more academic articles about the coronavirus than other news websites.
    Notice that more relavent sites such as `medarxiv.org` (a website for publishing academic medical papers that contains several analyses of the cornavirus) do not appear in this list even though they are very similar to `virological.org`.

1. Politics similarity:
    `breitbart.com` is a conservative news source that is well known for supporting President Trump.
    They published the following article:
    ```
    Pollak: Coronavirus Panic Partly Driven by Anti-Trump Hysteria
    ```
    We can run our inference algorithm using the command
    ```
    $ python3 names.py --infer --warm_start=models/gru_512x8 <<< "Pollak: Coronavirus Panic Partly Driven by Anti-Trump Hysteria"
    ```
    The top 5 predictions are:
    ```
       0 ussanews.com (0.04)
       1 www.infowars.com (0.04)
       2 crossman66.wordpress.com (0.04)
       3 fromthetrenchesworldreport.com (0.03)
       4 jonsnewplace.wordpress.com (0.02)
    ```
    In this case, Breitbart does not appear as one of the top predictions.
    But all the other sources listed share a similar conservative perspective.

<!--
koreatimes.co.kr
```
S. Korea reports 1st confirmed case of China coronavirus
Advice for vaccination against seasonal flu
North Korean newspaper urges 'absolute obedience' to Pyongyang's campaign to fight coronavirus
```
blogs.sciencemag.org
```
Covid-19 Small Molecule Therapies Reviewed
Coronavirus: Some Clinical Trial Data
What the Coronavirus Proteins Are Targeting
```
www.infowars.com
```
$ python3 names.py --infer --warm_start=models/gru_512x8 <<EOF
Drug Dealer: People Are “Panic Buying” Cocaine and Weed to Cope With Coronavirus Lockdown
Exiled Chinese Billionaire Claims 1.5 Million Infected With Coronavirus, 50,000 Dead
COVID-19 Relief Bill Totaling $2.2 Trillion Signed Into Law by Trump
Democrats Want To REVERSE Trump’s Travel Bans Despite Coronavirus Spread
EOF
```
www.huffpost.com
```
California Governor Calls Out What A Mess Coronavirus Testing Is Right Now
Wilbur Ross Roundly Ripped For Predicting Coronavirus Will Be Good For U.S. Jobs
Idris Elba Tests Positive For Coronavirus
```

**The problem:**
-->

**The Problem:**
We have a (crude) way to measure the similarity of two hostnames,
but we can't explain *why* two hostnames get measured similarly.
Your goal in this assignment is to find these explanations using the "sliding window algorithm".

### The sliding window algorithm

The sliding window algorithm is a folklore technique for explaining the result of any machine learning algorithm.
There are more sophisticated algorithms (such as [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap)),
but these are significantly more difficult to implement and interpret.
They do both have nice libraries, however, which give pretty visualizations.

**Basic idea.**
If we have an input sentence
```
Pollak: Coronavirus Panic Partly Driven by Anti-Trump Hysteria
```
and we want to know how important the word `Trump` is for our final classification,
we can:
(1) remove the word `Trump`,
(2) rerun the classification on the modified sentence,
and (3) compare the results of the model on the modified and unmodified sentences.
If the results are similar, then the word `Trump` is not important;
if the results are different, then the word `Trump` is important.

**How to remove a word?**
There are a surprising number of ways to remove a word from a sentence:

1. Create a new sentence by concatenating everything to the left and right of the removed word.
    Thus, our example sentence would become
    ```
    Pollak: Coronavirus Panic Partly Driven by Anti- Hysteria
    ```
    This method is easy to implement,
    but can result in grammatically incorrect sentences.
    Since our model is only trained on grammatically correct sentences,
    there is no reason to expect it to perform well on malformed sentences,
    and it is likely to output a large difference for every word in the sentence.

2. Replace the word with another word.
    In our example sentence, we might replace the word `Trump` with the word `Biden` to get
    ```
    Pollak: Coronavirus Panic Partly Driven by Anti-Biden Hysteria
    ```
    This sentence is now grammatically correct,
    and so we could expect our model to do reasonably on it.
    But how well it does would depend on our choice of replacement word,
    and so we would need to do many replacements and take an average to get a good estimate of the word's importance.

3. Insert "blank" inputs where the selected word should be.
    Recall that the inputs to our neural network are one-hot encoded letters.
    Therefore, every letter has a vector associated with it that has exactly one `1`.
    If we replace the `1` with a `0`, then the model will still know that there is a word in this location (because the size of the input tensor doesn't change),
    but the model is getting no information about what that word is.

**Comparing model outputs.**
There are many ways to calculate the similarity between two model outputs,
but the simplest is using the Euclidean distance between output the model probabilities,
and that's what you should use in this assignment.

**Example outputs.**
In the following images, each word is colored with the Euclidean distance calculated above.
Darker green values indicate that the word is more important.

<p align=center>
<img src=img/line0000.word.png width=800px>
</p>

<p align=center>
<img src=img/line0001.word.png width=800px>
</p>

<p align=center>
<img src=img/line0002.word.png width=800px>
</p>

**Pseudocode.**
The following pseudocode summarizes the sliding window explanation algorithm.
```
construct input_tensor from input_sentence
let probs = softmax(model(input_tensor))
for each word in the input sentence:
    construct input_tensor' by setting the colums associated with word to 0
    let probs' = softmax(model(input_tensor))
    weight[word] = |probs-probs'|
```

**Character level explanations.**
By repeating the above procedure with individual characters (rather than words),
we can generate character-level explanations of our text.

The resulting explanations look like:

<p align=center>
<img src=img/line0000.char.png width=800px>
</p>

<p align=center>
<img src=img/line0001.char.png width=800px>
</p>

<p align=center>
<img src=img/line0002.char.png width=800px>
</p>

<!--
Example outputs:
[Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/)
-->

### Tasks for you to complete

1. Follow the instructions above to download the pretrained model weights 
1. Implement the `explain` function in the `names.py` files source code
1. Reproduce the example explanations above to ensure that your code is working correctly
1. Select 3 titles from the training data and generate word/char level explanations for these titles

### Submission

Upload your explanation images and source code to sakai.

## Part 2: the attention mechanism and fine tuning

This part of the assignment is based on a new dataset [corona.multilan100.jsonl.gz](https://izbicki.me/public/cs/cs181/corona.multilang100.jsonl.gz),
which is in the same format as the previous dataset.
You should download this file and place it in your project folder.

Unlike the previous dataset, this dataset is multilingual.
It contains news headlines written in:
1. English,
1. Spanish,
1. Portuguese,
1. Italian,
1. French,
1. German,
1. Russian,
1. Chinese,
1. Korean,
1. and Japanese.

For each of these 10 languages, 10 prominent news sources were selected to be included in the dataset.
There are therefore 100 classes.
The overall size of the dataset 106767 headines, and so there are about 1000 headlines per news source.

The character level model we used before will not work in this multilingual setting because many of these languages use different vocabularies.
We will instead use the BERT transformer model and the [transformers](https://huggingface.co/transformers/) python library.

### Tasks for you to complete

1. Add support for training a multilingual BERT model to the `names.py` file.
1. Train the BERT model so that you get at least the following training accuuracies:
    1. accuracy@1 >= 0.3
    1. accuracy@20 >= 0.9

    These are very conservative numbers.
    You can view my [tensorboard.dev log](https://tensorboard.dev/experiment/2WJbkgdyTlGvh6Gk4mu0PQ/#scalars&_smoothingWeight=0.99) to see what type of performance levels are possible.

    You will have to experiment with different hyperparamter combinations in order to get good results.
    You do not have to do any warmstarts to get these results.
    I encourage you to try warmstarts since you can get much better accuracies,
    but the computational expense may be too much for some of your computers,
    and so I am not requiring it.

1. Generate a tensorboard.dev plot showing your model training progress

### Submission

Upload the link to you tensorboard.dev output on sakai

### Optional task

Extend the explanation code from part 1 so that it works on the BERT model as well.

## Part 3: embeddings

Coming soon...
