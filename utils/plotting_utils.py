import matplotlib.pyplot as plt
import numpy as np
print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def autolabel(rects,ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')

def plot_histogram(labelToCount, title):
	print("labelToCount",labelToCount)
	fig, ax = plt.subplots(figsize=(10, 6))
	width = 0.04 # width of the bars
	countList = [int(x) for x in list(labelToCount.keys())]
	#print("countList",countList)
	rects = ax.bar(countList, labelToCount.values())
	ax.set_ylabel('Count')
	ax.set_xlabel('Labels')
	ax.set_title(title)
	ax.set_xticks(countList)
	#ax.set_xticklabels(('Negative', 'Neutral', 'Positive'))
	autolabel(rects,ax)
	plt.show()


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