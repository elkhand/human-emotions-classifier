from __future__ import print_function
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import sys
import pprint
import pandas as pd
import numpy as np
import math
import os
from utils.scoring_utils import *
import nltk
# nltk.download('stopwords')
# nltk.data.path.append('~/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



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
    """read OASIS.csv into data frame"""
    return pd.read_csv(oasis_csv_path,header = 0, names=["id","theme","category","source",
                                                "valence_mean","valence_std", "valence_n",
                                               "arousal_mean","arousal_std", "arousal_n",]) #,  index_col="id"

def read_caption_csv_into_dataframe(caption_csv_path, delimeter=","):
    """read caption.csv into data frame"""
    return pd.read_csv(caption_csv_path, header=0, names=["id","image_title","caption",],  sep=delimeter) # index_col="id",   

def get_caption_to_label(caption_csv_path,delimeter=","):
    imageIdToCaption = get_image_id_to_caption(caption_csv_path,delimeter)
    imageIdToLabel = label_image_captions(imageIdToCaption)
    labelToCaption = {}
    for imageId in imageIdToCaption:
        caption = imageIdToCaption[imageId]
        label = imageIdToLabel[imageId]
        if label not in labelToCaption:
            labelToCaption[label] = []
        labelToCaption[label].append(caption)
    return labelToCaption


def get_image_id_to_valence_mean(oasis_csv_path):
    """read OASIS.csv into data frame and return {imageId : valence_mean}"""
    valence_df = read_oasis_csv_into_dataframe(oasis_csv_path)
    return dict(zip(valence_df.id, valence_df.valence_mean))

def get_image_title_to_image_id(oasis_csv_path):
    """read OASIS.csv into data frame and return {imageTitle : imageId}"""
    oasis_df = read_oasis_csv_into_dataframe(oasis_csv_path)
    return dict(zip(oasis_df.theme,oasis_df.id))

def get_image_id_to_caption(caption_csv_path,delimeter=","):
    """read caption csv into dataframe and return {imageid : caption}"""
    caption_df = read_caption_csv_into_dataframe(caption_csv_path, delimeter)
    imageIdToCaption = dict(zip(caption_df.id, caption_df.caption))
    # Transform imageId to be compliant with OASIS imageId, append "I" to the beginning
    imageIdToCaptionTransformed = {}
    for imageId in imageIdToCaption:
        caption = imageIdToCaption[imageId]
        newImageId = str(imageId)
        if "I" not in str(imageId):
            newImageId = "I" + str(imageId)
        imageIdToCaptionTransformed[newImageId] = caption 
    return imageIdToCaptionTransformed

def label_image_captions(imageIdToCaption):
    """given {imageId : caption}, return {imageId : label} using VADER"""
    imageIdToLabel = {}
    skipCount = 0
    for imageId in imageIdToCaption:
        caption = imageIdToCaption[imageId]
        # Skip missing captions
        if isinstance(caption, float) and math.isnan(caption):
            #print("Skipped image ",imageId," because caption did not exist")
            skipCount += 1
            continue
        label = get_label_for_caption_via_vader(caption)
        imageIdToLabel[imageId] = label
    print("Total skipped images", skipCount)
    return imageIdToLabel

def get_vader_compound_scores(imageIdToCaption):
    """this function returns vader scores for histogram plotting"""
    vaderScores = []
    for imageId in imageIdToCaption:
        caption = imageIdToCaption[imageId]
        vaderScores.append(get_vader_compound_score_per_caption(caption))
    return vaderScores

def get_vader_compound_score_per_caption(caption):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(caption)
    return scores['compound']

def get_label_for_caption_via_vader(caption):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(caption)
    label = convert_compound_score_to_label(scores['compound'])
    return label

def clean_auto_generated_captions(inputPathToAutoGenCaptions, outputPathToAutoGenCaptions, oasis_csv_path):
    """This function will clean/format auto generated captions in inputPathToAutoGenCaptions and 
    store in  outputPathToAutoGenCaptions in this format (imageId, image title,caption)"""
    delimeter = "|"
    imageTitleToImageId = get_image_title_to_image_id(oasis_csv_path)
    #print(imageTitleToImageId.keys())
    if ".txt" in outputPathToAutoGenCaptions:
        raise ValueError("You tried to delete input file by mistake, the output file should end with .csv")
    delete_file_if_exists(outputPathToAutoGenCaptions)

    with open(outputPathToAutoGenCaptions, 'a') as f:
        f.write("id,image_file,caption"+"\n")
    #printCnt = 0
    with open(inputPathToAutoGenCaptions, 'r') as f:
        for imageTitleLine in f:
            #print(imageTitleLine)
            if ")" in imageTitleLine or "(" in imageTitleLine:
                raise ValueError("Unexpected imageTitle line", imageTitleLine)
            cnt = 0
            imageTitle = get_image_title_from_image_title_line(imageTitleLine)
            #print(imageTitle)
            maxCaptionScore = -1
            maxCaptionLength = -1
            bestCaption = ""
            for captionLine in f:
                #print(captionLine)
                if str(cnt) not in captionLine:
                    raise ValueError("Unexpected caption line", captionLine, "cnt", cnt)
                cnt += 1
                captionScore = get_probability_from_caption_line(captionLine)
                caption = get_caption_from_caption_line(captionLine)
                if captionScore > maxCaptionScore:
                    maxCaptionScore = captionScore
                    maxCaptionLength = len(caption)
                    bestCaption = caption
                elif math.isclose(captionScore, maxCaptionScore, rel_tol=0.00001) and len(caption) > maxCaptionLength:
                    maxCaptionScore = captionScore
                    maxCaptionLength = len(caption)
                    bestCaption = caption
                if cnt == 3:
                    break 
            imageTitleWithoutExtension = imageTitle[:imageTitle.find(".")]
            imageId = imageTitleToImageId[imageTitleWithoutExtension]
            write_auto_generated_best_captions(imageId, imageTitle, bestCaption, delimeter, outputPathToAutoGenCaptions)
            #print("best: ",bestCaption, maxCaptionScore)
            #printCnt += 1
            #if printCnt > 10:
            #    break

def delete_file_if_exists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def get_probability_from_caption_line(captionLine):
    """this function extracts probability of caption from caption line"""
    pEqualStr = "p="
    closingPar = ")"
    score = captionLine.split(pEqualStr)[1]
    score = score.replace(closingPar, "") # get rid of ")"
    score = float(score)
    return score

def get_caption_from_caption_line(captionLine):
    """extract caption from caption and probability line"""
    #captionLine = captionLine.replace("<S>","")
    startClosingPar = captionLine.find(")")
    dotIndex = captionLine.find(".")
    if dotIndex == -1 or dotIndex > captionLine.find("<S>"):
        dotIndex = captionLine.find("<S>")
        #print(captionLine,dotIndex)
    caption = captionLine[startClosingPar+1:dotIndex].strip() + "."
    #if "<S>" in caption:
    #    print(captionLine,dotIndex, caption)
    return caption

def get_image_title_from_image_title_line(imageTitleLine):
    """Extract image title/name from imageTitleLine"""
    prefixStr = "Captions for image"
    colon = ":"
    imageTitle = imageTitleLine[imageTitleLine.find(prefixStr) + len(prefixStr): imageTitleLine.find(colon)].strip()
    return imageTitle

def write_auto_generated_best_captions(imageId, imageTitle, bestCaption, delimeter,  outputPathToAutoGenCaptions):
    """this function writes imageId, imageTitle, and the best caption into output file"""
    with open(outputPathToAutoGenCaptions, 'a') as f:
        f.write(imageId + delimeter + imageTitle + delimeter + bestCaption + "\n")

def calcualte_mean_caption_length(captionList):
    """"given captions, calculates avg caption lengths in words"""
    stop_words = set(stopwords.words('english'))
    wordCount = 0
    for caption in captionList:
        words = word_tokenize(caption)
        for word in words:
            if word in stop_words:
                continue
            wordCount += 1
    return wordCount/len(captionList)        