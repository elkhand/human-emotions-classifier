from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
print(__doc__)

import itertools
from collections import OrderedDict

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from PIL import Image

#import hecutils.scoring_utils as sc
#import hecutils.data_utils as dt

def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')

def print_label_to_count(labelToCount):
    labelToCount = OrderedDict(sorted(labelToCount.items()))
    print("labelToCount :",labelToCount)

def plot_histogram(labelToCount, title):
    print_label_to_count(labelToCount)
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.04 # width of the bars
    countList = [int(x) for x in list(labelToCount.keys())]
    rects = ax.bar(countList, labelToCount.values())
    ax.set_ylabel('Count')
    ax.set_xlabel('Labels')
    ax.set_title(title)
    ax.set_xticks(countList)
    #ax.set_xticklabels(('Negative', 'Neutral', 'Positive'))
    autolabel(rects,ax)
    plt.show()


def image_label_histogram(oasis_csv_path, neutralLow,neutralHigh):
    import hecutils.data_utils as dt
    import hecutils.scoring_utils as sc
    imageIdToLabel = {}
    imageIdToValence = dt.get_image_id_to_valence_mean(oasis_csv_path)
    valence_values = list(imageIdToValence.values())
    minValence = min(valence_values)
    maxValence = max(valence_values)
    meanValence = np.mean(valence_values)
    medianValence = np.median(valence_values)
    stdValence = np.std(valence_values)
    print("Stats of valence scores\n--------------------------------","\nminValence", minValence,
          "\nmaxValence",maxValence, "\nmeanValence",meanValence,
          "\nstdValence", stdValence,"\nmedianValence",medianValence,"\n--------------------------------\n")

    labelToCount = {}
    isStrResult = False
    # neutralLow=2.0
    # neutralHigh=4.0
    #neutralLow=3.0
    #neutralHigh=5.0
    for imageId in imageIdToValence:
        valenceScore = imageIdToValence[imageId]
        label = sc.evaluate_score(valenceScore, isStrResult, neutralLow, neutralHigh)
        imageIdToLabel[imageId] = label
        if label not in labelToCount:
            labelToCount[label] = 0
        labelToCount[label] += 1
    
    print("Total images",len(imageIdToValence.keys()))
    title = 'True Labels of OASIS based on valence score'
    plot_histogram(labelToCount, title)

    return [labelToCount, imageIdToLabel, imageIdToValence]

def get_label_count(imageIdToLabel):
	"""input is imageIdToLabel, returns {label: count}"""
	labelToCount = {}
	for imageId in imageIdToLabel:
		label = imageIdToLabel[imageId]
		if label not in labelToCount:
			labelToCount[label] = 1
		else:
			labelToCount[label] += 1
	return labelToCount


def plot_confusion_matrix_from_labels(trueLabels,predictedLabels, titleOfConfusionMatrix):
	"""plot confusion matrix"""
	#print("true labels",trueLabels)
	#print("predicted labels",predictedLabels)
	class_names = ['negative', 'neutral', 'positive'] # [-1, 0 , 1]
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(trueLabels, predictedLabels)
	#print("cnf_matrix",cnf_matrix)
	np.set_printoptions(precision=2)
	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title=titleOfConfusionMatrix)
	plt.show()




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 
def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10)):
    """source: https://stackoverflow.com/questions/36006136/how-to-display-images-in-a-row-with-ipython-display"""
    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        #plt.imshow(list_of_images[i])
        pil_im = Image.open(list_of_images[i], 'r')
        plt.imshow(pil_im)
        plt.axis('off')
        if len(list_of_titles) >= len(list_of_images):
            plt.title(list_of_titles[i])    