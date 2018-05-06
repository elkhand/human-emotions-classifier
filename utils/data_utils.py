from __future__ import print_function
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.data_utils import *

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
