# CSCI181: Deep Learning

## About the Instructor

|||
|-|-|
| Name | Mike Izbicki (call me Mike) |
| Email | mizbicki@cmc.edu |
| Office | Adams 216 |
| Office Hours | Monday 9:00-10:00AM, Tuesday/Thursday 2:30-3:30PM, or by appointment ([see my schedule](https://izbicki.me/schedule.html));<br/> if my door is open, feel free to come in |
| Webpage | [izbicki.me](https://izbicki.me) |
| Research | Machine Learning (see [izbicki.me/research.html](https://izbicki.me/research.html) for some past projects) |
| Fun Facts | grew up in San Clemente, 7 years in the navy, phd/postdoc at UC Riverside, taught in [DPRK](https://pust.co) |

## About the Course

This is a course on **deep learning** (not big data).

<p align=center>
<img src=img/layers.png width=600px>
</p>

**Course Objectives:**

Learning objectives:

1. Write basic PyTorch applications
1. Understand the "classic" deep network architectures
1. Use existing models in a "reasonable" way
1. Understand the limitations of deep learning
1. Read research papers published in deep learning
1. Understand what graduate school in machine learning is like
1. (Joke) [Understand that Schmidhuber invented machine learning](https://www.reddit.com/r/MachineLearning/comments/eivtmq/d_nominate_jurgen_schmidhuber_for_the_2020_turing/)

My personal goal:

1. Find students to conduct research with me

**Expected Background:**

Necessary:

1. Basic Python programming
1. Linear algebra
1. Calc III
1. Statistics

Good to have:

1. Machine learning / data mining
1. Lots of math
1. Familiarity with Unix and github

**Resources:**

Textbook:

1. [The Deep Learning Book](http://www.deeplearningbook.org/), by Ian Goodfellow and Yoshua Bengio and Aaron Courville; I will assume that you already know all of Part I of this book (basically the equivalent of a data mining/machine learning course)
1. Various papers/webpages as listed below

Deep learning examples:

1. Images / Video
    1. [Deoldify](https://github.com/jantic/DeOldify)
    1. [style transfer](https://genekogan.com/works/style-transfer/)
    1. [more style transfer](https://github.com/lengstrom/fast-style-transfer)
    1. [dance coreography](https://experiments.withgoogle.com/billtjonesai)
    1. [StyleGAN](https://github.com/NVlabs/stylegan)
    1. [DeepPrivacy](https://github.com/hukkelas/DeepPrivacy)
    1. https://thispersondoesnotexist.com/
    1. https://thiscatdoesnotexist.com/
    1. [Deep fakes](https://www.creativebloq.com/features/deepfake-examples)
    1. [In Event of Moon Disaster](https://www.wbur.org/news/2019/11/22/mit-nixon-deep-fake)

1. Text
    1. [Image captioning](https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/)
    1. [AI Dungeon](https://www.aidungeon.io/)
    1. https://www.thisstorydoesnotexist.com/
    1. https://translate.google.com

1. Games
    1. [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far)
    1. [Dota 2](https://openai.com/projects/five/)
    1. [StarCraft 2](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning)
    1. [MarioCart](https://www.youtube.com/watch?v=Ipi40cb_RsI)
    1. [Mario](https://www.youtube.com/watch?v=qv6UVOQ0F44)

1. Other
    1. [iSketchNFill](https://github.com/arnabgho/iSketchNFill)
    1. [scrying-pen](https://experiments.withgoogle.com/scrying-pen)
    1. [Tacotron](https://google.github.io/tacotron/publications/speaker_adaptation/)

The good:

1. [most influential research in 2019 is deep learning papers](https://www.altmetric.com/top100/2019/)
1. [/r/machinelearning](https://reddit.com/r/machinelearning)
1. [recent open source AI programs](https://www.reddit.com/r/MachineLearning/comments/egyp7w/d_what_is_your_favorite_opensource_project_of/)
1. [The state of jobs in deep learning](https://www.reddit.com/r/MachineLearning/comments/egt6dp/d_are_decent_machine_learning_graduates_having_a/)
1. [The decade in review](https://leogao.dev/2019/12/31/The-Decade-of-Deep-Learning/)

The bad:

1. [Machine learning reproducibility crisis](https://www.wired.com/story/artificial-intelligence-confronts-reproducibility-crisis/)
1. [Logistic regression vs deep learning in aftershock prediction](https://www.reddit.com/r/MachineLearning/comments/dcy2ar/r_one_neuron_versus_deep_learning_in_aftershock/)
1. [Pictures of black people](https://news.ycombinator.com/item?id=21147916)
1. [NLP Clever Hans BERT](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/)
1. [Ex-Baidu researcher denies cheating at machine learning competition](https://www.enterpriseai.news/2015/06/12/baidu-fires-deep-images-ren-wu/)

Computing resources:

1. [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb) provides 12 hours of free GPUs in a Jupyter notebook
1. [Kaggle](https://forums.fast.ai/t/kaggle-kernels-now-support-gpu-for-free/16217) provides 30 hours of free GPU
1. I have a 40CPU/8GPU machine that you can access for the course
1. I have another 4CPU/1GPU machine that needs someone to set it up

Videos:

1. [3blue1brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
1. [2 minute papers](https://www.youtube.com/user/keeroyz)
1. [arxiv insights](https://www.youtube.com/channel/UCNIkB2IeJ-6AmZv7bQ1oBYg)

<!--
FIXME:
[What's wrong with notebooks](http://web.eecs.utk.edu/~azh/blog/notebookpainpoints.html)
-->

## Schedule

| Week | Date         | Topic                                   |
| ---- | ------------ | --------------------------------------  |
| 1    | Tues, 21 Jan | Intro: Examples of Deep Learning        |
| 1    | Thur, 23 Jan | Automatic differentiation<br><ul><li>[pytorch tutorial part 1](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)</li><li>[pytorch tutorial part 2](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)</li><li>[automatic differentiation tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)</li><li>[einstein summation tutorial](https://rockt.github.io/2018/04/30/einsum)</li><li>[NeurIPS paper](http://papers.nips.cc/paper/8092-automatic-differentiation-in-ml-where-we-are-and-where-we-should-be-going)</li><li>[JMLR paper](http://www.jmlr.org/papers/v18/17-468.html)</li><li>[pytoch: forward mode ad](https://github.com/pytorch/pytorch/issues/10223)</li><li>[tensorflow: forward mode ad](https://github.com/pytorch/pytorch/issues/10223)</li></ul> |
| 2    | Tues, 28 Jan | Machine Learning Basics (Deep Learning Book Part 1, especially chapters 5.2-5.4)                 |
| 2    | Thur, 30 Jan | Optimization<br><ul><li>[why momentum really works](https://distill.pub/2017/momentum/)</li><li>[Leon Bottou's SGD paper](https://datajobs.com/data-science-repo/Stochastic-Gradient-Descent-[Leon-Bottou].pdf)</li><li>[pytorch loss functions](https://pytorch.org/docs/stable/nn.html#crossentropyloss)</li><li>[reflections on random kitchen sinks](http://www.argmin.net/2017/12/05/kitchen-sinks/)</li><li>[Ali Rahimi's NIPS/NeurIPS 2017 keynote](https://www.youtube.com/watch?v=Qi1Yry33TQE)</li><li>[OpenAI switches to PyTorch](https://openai.com/blog/openai-pytorch/)</li></ul>                            |
| 3    | Tues, 04 Feb | Image: CNNs<br><ul><li>[Stanford lecture slides](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf)</li></ul>                             |
| 3    | Thur, 06 Feb | Image: CNNs II<br><ul><li>[An intuitive explanation of CNNs](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) (compare with [eigenfaces](https://towardsdatascience.com/eigenfaces-recovering-humans-from-ghosts-17606c328184))</li><li>[The history of neural networks](https://dataconomy.com/2017/04/history-neural-networks/)</li></ul>[Summer Research](https://www.cmc.edu/summer-research/program-overview) |
| 4    | Tues, 11 Feb | Regularization                          |
| 4    | Thur, 13 Feb | Image: ResNet<br><ul><li>[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)</li><li>[An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)</li><li>[CVPR2016 Video](https://www.youtube.com/watch?v=C6tLw-rPQ2o)</li></ul>More links:<ul><li>[Schmidhuber on ResNet I](http://people.idsia.ch/~juergen/microsoft-wins-imagenet-through-feedforward-LSTM-without-gates.html)</li><li>[Schmidhuber on ResNet II](http://people.idsia.ch/~juergen/highway-networks.html)</li><li>[Baidu scandal at ILSVRC15](https://web.archive.org/web/20150602165531/http://www.image-net.org/challenges/LSVRC/announcement-June-2-2015)</li><li>[What I learned from competing against a convnet on ImageNet](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)</li><li>[MSCOCO](http://cocodataset.org)</li></ul>                           |
| 5    | Tues, 18 Feb | ResNet continued                        |
| 5    | Thur, 20 Feb | ResNet continued                        |
| 6    | Tues, 25 Feb | YOLO<br><ul><li>[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)</li><li>[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) and [reviews](https://pjreddie.com/publications/yolo9000/)</li><li>[You Only Look Once: Unified Real-Time Object Detection](http://arxiv.org/abs/1506.02640) and [reviews](https://pjreddie.com/publications/yolo/)</li><li>[YOLO video example](https://www.youtube.com/watch?v=MPU2HistivI)</li><li>[YOLO video presentation](https://www.youtube.com/watch?v=NM6lrxy0bxs&feature=youtu.be)</li><li>[Joseph Redmon's CV](https://pjreddie.com/static/Redmon%20Resume.pdf)</li></ul>The MvMF loss for geolocation:<ul><li>[ICML-PKDD paper](https://izbicki.me/public/papers/ecmlpkdd2019-image-geolocation.pdf)</li></ul>                           |
| 6    | Thur, 27 Feb | Text: Word vs Character Models          |
| 7    | Tues, 03 Mar | Text: RNN                               |
| 7    | Thur, 05 Mar | Text: LSTM/GRU                          |
| 8    | Tues, 10 Mar | Text: Attention                         |
| 8    | Thur, 12 Mar | Text: Transformers (paper, [blog post](http://jalammar.github.io/illustrated-transformer/)) |
| 9    | Tues, 17 Mar | Text: Translation                       |
| 9    | Thur, 19 Mar | **NO CLASS:** Spring Break              |
| 10   | Tues, 24 Mar | **NO CLASS:** Spring Break              |
| 10   | Thur, 26 Mar | TBD                                     |
| 11   | Tues, 31 Mar | TBD                                     |
| 11   | Thur, 02 Apr | TBD                                     |
| 12   | Tues, 07 Apr | TBD                                     |
| 12   | Thur, 09 Apr | TBD                                     |
| 13   | Tues, 14 Apr | TBD                                     |
| 13   | Thur, 16 Apr | TBD                                     |
| 14   | Tues, 21 Apr | TBD                                     |
| 14   | Thur, 23 Apr | TBD                                     |
| 15   | Tues, 28 Apr | TBD                                     |
| 15   | Thur, 30 Apr | Project Presentations                   |
| 16   | Thur, 05 May | Project Presentations                   |
| 16   | Thur, 07 May | **NO CLASS:** Reading Day               |

<!--
| 3    | Thur, 06 Feb | Regularization: Dropout ([JMLR](http://jmlr.org/papers/v15/srivastava14a.html), [pytorch](https://stackoverflow.com/questions/53419474/using-dropout-in-pytorch-nn-dropout-vs-f-dropout)) |
| 4    | Tues, 11 Feb | Regularization: BatchNorm ([ICML](http://proceedings.mlr.press/v37/ioffe15.html), [pytorch](https://stackoverflow.com/questions/47197885/how-to-do-fully-connected-batch-norm-in-pytorch))              |
-->

<!-- GoogLeNet: https://arxiv.org/abs/1409.4842 -->
<!-- MAY 8: Senior Grades due -->

### Assignments

| Week | Weight | Topic                           |
| ---- | ------ | ------------------------------- |
| 2    | 10     | Rosenbrock Function             |
| 3    | 10     | Crossentropy Loss               |
| 4    | 10     | CNN                             |
| 6    | 10     | Image Transfer Learning         |
| 7    | 10     | RNN                             |
| 10   | 10     | Text Transfer Learning          |
| --   | 10     | Reading                         |
| 15   | 30     | Project                         |

There are no exams in this course.

**Late Work Policy:**

You lose 10% on the assignment for each day late.
If you have extenuating circumstances, contact me in advance of the due date and I may extend the due date for you.

**Collaboration Policy:**

You are encouraged to work together with other students on all assignments and use any online resources.
Learning the course material is your responsibility,
and so do whatever collaboration will help you learn the material.

<!--
Images:
Transfer Learning
Style Transfer (https://github.com/chuanli11/CNNMRF)
Object Detection
Image Captioning
Deep Dream
Deep Fake

Inception History:
https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

Cross Entropy
CNN
Data Augmentation

MNIST vs CIFAR vs ImageNet

L2 Regularization
Dropout
Batch Norm
ResNet

RNN
LSTM
GRU
ELMO
BERT
Transformer

https://github.com/taehoonlee/tensornets

Text Transfer Learning (1)
https://github.com/NVIDIA/sentiment-discovery/
Maybe use this instead? https://www.reddit.com/r/MachineLearning/comments/d5x71d/p_gobbli_a_python_framework_for_text/

Style transfer in text:
https://github.com/fuzhenxin/Style-Transfer-in-Text

GPT-2 fine tuning
https://github.com/minimaxir/gpt-2-simple
GPT-2 plays chess: https://news.ycombinator.com/item?id=21980924

-->

<!--
## Self Grading

[An outlook on self-assessment of homework assignments in higher mathematics education](https://link.springer.com/article/10.1186/s40594-018-0146-z)

Also *Your* Job to Learn! Helping Students Reflect on their Learning Progress

Should you Allow your Students to Grade their own Homework?

Peer and Self Assessment in Massive Online Classes
-->

## Accommodations for Disabilities

I want you to succeed and I'll make every effort to ensure that you can.
If you need any accommodations, please ask.

If you have already established accommodations with Disability Services at CMC, please communicate your approved accommodations to me at your earliest convenience so we can discuss your needs in this course. You can start this conversation by forwarding me your accommodation letter. If you have not yet established accommodations through Disability Services, but have a temporary health condition or permanent disability (conditions include but are not limited to: mental health, attention-related, learning, vision, hearing, physical or health), you are encouraged to contact Assistant Dean for Disability Services & Academic Success, Kari Rood, at disabilityservices@cmc.edu to ask questions and/or begin the process. General information and the Request for Accommodations form can be found at the CMC DOS Disability Service’s website. Please note that arrangements must be made with advance notice in order to access the reasonable accommodations. You are able to request accommodations from CMC Disability Services at any point in the semester. Be mindful that this process may take some time to complete and accommodations are not retroactive. It is important to Claremont McKenna College to create inclusive and accessible learning environments consistent with federal and state law. If you are not a CMC student, please connect with the Disability Services Coordinator on your campus regarding a similar process.

