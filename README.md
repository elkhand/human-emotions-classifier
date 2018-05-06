# human-emotions-classifier
CS231n Final project

## Problem
I’m interested in solving, given an image, (binary) classifying the emotion the image evokes has positive or a negative valence (affective response) on viewer person. I’ll be evaluating image pixels as a feature, caption as a feature, and both features together, how they are affecting the accuracy of the model.

Writeup (draft version): https://docs.google.com/document/d/1P8DjZgdnyYcABErRjv0Vk2UZL0CX2cWWogv5IQyjCK4/edit#

## Dataset
Download link: https://www.dropbox.com/sh/4qaoqs77c9e5muh/AABBw07ozE__2Y0LVQHVL-8ca?dl=0 

I’ll be using Open Affective Standardized Image Set(OASIS) dataset for this final project. OASIS dataset is an open-access online stimulus set containing 900 color images depicting a broad spectrum of themes, including humans, animals, objects, and scenes, along with normative ratings on two affective dimensions—valence (i.e., the degree of positive or negative affective response that the image evokes) and arousal (i.e., the intensity of the affective response that the image evokes)[1]

Captions link: http://zmana-caption.herokuapp.com/captioner/ 

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
