import nltk
from nltk.corpus import stopwords
from keras.layers.core import  ActivityRegularization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.core import Flatten
from keras.layers import Activation, TimeDistributed, Embedding, GRU, Bidirectional, LSTM
from sklearn.metrics import f1_score
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import time
stop_words = set(stopwords.words('english'))

def get_word_embedding(wordToVec, word, config):
    word = word.lower()
    if word in wordToVec:
        return wordToVec[word]
    else:
        return np.zeros(config['embedding_dimension'],)
    
def get_sequence_embedding(wordToVec, words, max_seq_len, config):
    if len(words) <= max_seq_len:
        # Add padding
        x_seq = np.array([get_word_embedding(wordToVec, word, config) for word in words])
        x_seq = np.lib.pad(x_seq, ((0,max_seq_len-x_seq.shape[0]),(0,0)), 'constant')
    else:
        x_seq = []
        for i in range(max_seq_len):
            x_seq.append(get_word_embedding(wordToVec, words[i], config))
        x_seq = np.array(x_seq)
    return x_seq
        
def convert_index_to_one_hot(y_train_index, num_of_classes):
    y_train = np.zeros((y_train_index.shape[0],num_of_classes))
    y_train[range(y_train_index.shape[0]),y_train_index] = 1
    return y_train


def generate_model_name(filename, best_acc_val):
    timestamp = str(time.time()).split(".")[0]
    best_acc_val = round(best_acc_val,4)
    filename += "-" + str(best_acc_val) + "-" + timestamp
    return filename

def get_words_withoutstopwords(words):
    words_without_stopwords = []
    for word in words:
        if word not in stop_words:
            words_without_stopwords.append(word)
    return words_without_stopwords

def load_dataset_StratifiedKFold(dfKFold, wordToVec, max_seq_len, class_to_index, index_to_class, config):
    sentences = []
    label_index = []
    num_of_classes = 0
    for index, row in dfKFold.iterrows():
        caption = row['caption']
        label = row['label']
        words = caption.split(" ")
        words = get_non_stop_words(words)
        sentence_embedding = get_sequence_embedding(wordToVec, words, max_seq_len, config)
        sentences.append(sentence_embedding)
        if label in class_to_index:
            label_index.append(class_to_index[label])
        else:
            num_of_classes += 1
            class_to_index[label] = num_of_classes - 1
            index_to_class[num_of_classes - 1] = label
            label_index.append(class_to_index[label])
    X_train = np.array(sentences)
    y_train = np.array(label_index)
    return (X_train, y_train, num_of_classes, class_to_index, index_to_class)


def get_non_stop_word_count(words):
    count = 0
    for word in words:
        if word not in stop_words:
            count += 1
    return count


def get_non_stop_words(words):
    non_stop_words = []
    for word in words:
        if word not in stop_words:
            non_stop_words.append(word)
    return non_stop_words

def change_label_str_to_int(labelStr):
    if labelStr == "negative":
        return -1
    elif  labelStr == "neutral":
        return 0
    elif  labelStr == "positive":
        return 1

def get_label_map_from_train_set(dfInput, wordToVec, max_seq_len, config):
    class_to_index = {}
    index_to_class = {}
    _, _, num_of_classes, class_to_index, index_to_class = \
            load_dataset_StratifiedKFold(
                            dfInput,
                            wordToVec, 
                            max_seq_len, 
                            class_to_index, 
                            index_to_class,
                            config)
    return  (num_of_classes, class_to_index, index_to_class)