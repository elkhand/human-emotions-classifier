from __future__ import print_function
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import csv
import sys
import pprint
import pandas as pd
import numpy as np

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


def print_polarity_scores(image_captions):
    analyzer = SentimentIntensityAnalyzer()
    captionToLabels = {}
    for caption in image_captions:
        scores = analyzer.polarity_scores(caption)
        label = convert_compound_score_to_string(scores['compound'])
        print("{} \n {} {}\n".format(caption, str(scores),label))
        label = convert_compound_score_to_label(scores['compound'])
        captionToLabels[caption] = label
    return captionToLabels        


def read_oasis_csv_into_dataframe(oasis_csv_path):
	return pd.read_csv(oasis_csv_path,header = 0, names=["id","theme","category","source",
                                                "valence_mean","valence_std", "valence_n",
                                               "arousal_mean","arousal_std", "arousal_n",])

def get_image_id_to_valence_mean(oasis_csv_path):
	valence_df = read_oasis_csv_into_dataframe(oasis_csv_path)
	return dict(zip(valence_df.id, valence_df.valence_mean))


def get_normalized_valence_mean(imageIdToValence):
	""""convert ragen of valence mean into new range of [-1,1]"""
	# https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratio
	valence_values = list(imageIdToValence.values())
	oldMin = min(valence_values)
	oldMax = max(valence_values)
	newMax = 1
	newMin = -1
	oldRange = (oldMax - oldMin)
	new_valence_values = {}
	for imageId in imageIdToValence:
		oldValue = imageIdToValence[imageId]
		newValue = -100
		if (oldRange == 0):
			newValue = newMin
		else:
			newRange = (newMax - newMin)  
			newValue = (((oldValue - oldMin) * newRange) / oldRange) + newMin
		new_valence_values[imageId] = newValue
	return new_valence_values

