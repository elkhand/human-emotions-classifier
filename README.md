# human-emotions-classifier
CS231n Final project

## Problem
I’m interested in solving, given an image, (binary) classifying the emotion the image evokes has positive or a negative valence (affective response) on viewer person. I’ll be evaluating image pixels as a feature, caption as a feature, and both features together, how they are affecting the accuracy of the model.

Writeup (draft version): https://docs.google.com/document/d/1P8DjZgdnyYcABErRjv0Vk2UZL0CX2cWWogv5IQyjCK4/edit#

## Dataset
Download link: http://www.benedekkurdi.com/#oasis

I’ll be using Open Affective Standardized Image Set(OASIS) dataset for this final project. OASIS dataset is an open-access online stimulus set containing 900 color images depicting a broad spectrum of themes, including humans, animals, objects, and scenes, along with normative ratings on two affective dimensions—valence (i.e., the degree of positive or negative affective response that the image evokes) and arousal (i.e., the intensity of the affective response that the image evokes)[1]

Captions link: http://zmana-caption.herokuapp.com/captioner/ 

### Auto Generated Captions using Show and Tell: A Neural Image Caption Generator

Git repo: https://github.com/tensorflow/models/tree/master/research/im2txt

A TensorFlow implementation of the image-to-text model described in the paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge."

Full text available at: http://arxiv.org/abs/1609.06647

The Show and Tell model is an example of an encoder-decoder neural network. It works by first "encoding" an image into a fixed-length vector representation, and then "decoding" the representation into a natural language description.

The image encoder is a deep convolutional neural network. This type of network is widely used for image tasks and is currently state-of-the-art for object recognition and detection. Our particular choice of network is the Inception v3 image recognition model pretrained on the ILSVRC-2012-CLS image classification dataset.

The decoder is a long short-term memory (LSTM) network. This type of network is commonly used for sequence modeling tasks such as language modeling and machine translation. In the Show and Tell model, the LSTM network is trained as a language model conditioned on the image encoding.

Words in the captions are represented with an embedding model. Each word in the vocabulary is associated with a fixed-length vector representation that is learned during training.

![Show and Tell Architecture](show_and_tell_architecture.png)

Auto generated captions can be found here:

```
dataset/metadata/auto_generated_captions.txt
```

## Setup instructions

This setup instructions will resemble CS231N setup instructions: http://cs231n.github.io/setup-instructions/ 

Installing Anaconda: The free Anaconda Python distribution is recommended, which provides an easy way for you to handle package dependencies. Please be sure to download the Python 3 version, which currently installs Python 3.6.

Anaconda Virtual environment: Once you have Anaconda installed, it makes sense to create a virtual environment for the course. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run (in a terminal)

```
conda create -n cs231n python=3.6 anaconda
```

to create an environment called **cs231n**.

Then, to activate and enter the environment, run
```
conda activate cs231n
```

To exit, you can simply close the window, or run
```
conda deactivate
```

Note that every time you want to work on this project, you should run *conda activate cs231n*.

After installing Anaconda, creating the virtual environment, activating that environment, then cd to the project root directory, and run the below command to install project dependencies:
```
pip install -r requirements.txt  # Install dependencies
```

Now you can install PyTorch:
```
conda install pytorch torchvision -c pytorch
```


Now you can install VADER:
```
pip install vaderSentiment
```

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. 

VADER source code: https://github.com/cjhutto/vaderSentiment 

## Download OASIS dataset
Inside project root directory:
```
cd dataset
./get_datasets.sh
```


## References
[1] Benedek Kurdi, Shayn Lozano, Mahzarin R. Banaji , Introducing the Open Affective Standardized Image Set (OASIS)
