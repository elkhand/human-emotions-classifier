from __future__ import print_function
#import hecutils.data_utils as dt
import hecutils.plotting_utils as pt
import math
import sys
from sklearn.metrics import f1_score



def evaluate_score(score, isStrResult, neutralLow, neutralHigh):
    if score < neutralHigh and score > neutralLow:
        return "NEUTRAL" if isStrResult  else 0
    elif score  < neutralLow or math.isclose(score, neutralLow, rel_tol=0.00001):
        return "NEGATIVE" if isStrResult    else -1
    elif score > neutralHigh or math.isclose(score, neutralHigh, rel_tol=0.00001):
        return "POSITIVE" if isStrResult    else  1
    else:
        errMsg = 'could not evaluate score', score,"neutralLow",neutralLow,"neutralHigh",neutralHigh
        raise ValueError(errMsg)
        sys.exit(errMsg)

def convert_compound_score_to_string(compound_score):
    compound_score *= 100
    if compound_score < 5 and compound_score > -5:
        return "NEUTRAL"
    elif compound_score  < -5:
        return "NEGATIVE"
    elif compound_score > 5:
        return "POSITIVE"

def convert_compound_score_to_label(compound_score):    
    """0 is for Neutral, 1 is for Positive, -1 is for Negative label """
    compound_score *= 100
    if compound_score < 5 and compound_score > -5:
        return 0
    elif compound_score  < -5:
        return -1
    elif compound_score > 5:
        return 1

def get_accuracy(imageIdToLabel, imageIdToLabelFromCaptions, titleOfConfusionMatrix):
    """compares labels calcualted from OASIS.csv valence_mean score vs labels calcualted via VADER from caption.csv"""
    posCorrect = 0 
    negCorrect = 0
    neutCorrect = 0
    totalCorrect = 0
    totalPos = 0
    totalNeg = 0
    totalNeut = 0
    total = len(imageIdToLabelFromCaptions.keys())
    f1Dict={}
    #f1Dict["TP"] =
    for imageId in imageIdToLabelFromCaptions.keys():
        #print("imageId",imageId)
        labelFromCaption = imageIdToLabelFromCaptions[imageId]
        trueLabel = imageIdToLabel[imageId]
        incr = compare_labels(labelFromCaption,trueLabel)
        #if incr == 0:
        #    print(imageId)
        totalCorrect += incr
        if labelFromCaption == 1:
            totalPos += 1
            posCorrect += incr
        elif labelFromCaption == 0:
            totalNeut += 1
            neutCorrect += incr
        elif labelFromCaption == -1:
            totalNeg += 1
            negCorrect += incr
    y_true, y_pred = get_labels(imageIdToLabel, imageIdToLabelFromCaptions)
    f1Score = f1_score(y_true, y_pred, average=None)
    pt.plot_confusion_matrix_from_labels(y_true, y_pred, titleOfConfusionMatrix)
    #print("labels in f1 score are in this order [negative, neutral, positive]")
    # labels in f1 score are in this order [negative, neutral, positive]
    result = { "total_accuracy": totalCorrect*100/total ,"pos_accuracy":posCorrect*100/totalPos,
             "neg_accuracy": negCorrect*100/totalNeg, "neutral_accuracy": neutCorrect*100/totalNeg}
    result["f1_score_neg"] = f1Score[0]
    result["f1_score_neut"] = f1Score[1]
    result["f1_score_pos"] = f1Score[2]
    return result

def get_labels(trueImageIdToLabel, imageIdToLabelFromCaptions):
    y_true = []
    y_pred = []
    for imageId in trueImageIdToLabel:
        y_true.append(trueImageIdToLabel[imageId])
        y_pred.append(imageIdToLabelFromCaptions[imageId])
    return [y_true, y_pred]

def compare_labels(label1, label2):
    if label1 == label2:
        return 1
    else:
        return 0

