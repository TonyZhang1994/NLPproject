
# -*- coding: utf-8 -*-
'''storage the text into a dict with the name being keys and recipe sentences being the content, the dict name is Recipe
    download some corpus and use it, ingredients, vegetables, meats'''





'''把 word vector 想办法放进feature里面  另外将feature standardized, 训练Word2vector的模型，可以做出来相似的菜谱有哪些？'''
import nltk
import re
import numpy as np
import os
import sklearn

def get_features(sentences):
    sentences = sentences.decode('utf-8')
    ftrs = []
    sents = nltk.sent_tokenize(sentences)
    feas = {}
    for key in corpus.keys():
        if key != "verbs":
            feas[key] = []
        else:
            feas[key] = 0.0
    time_needed = 0.0    # unit miniute, if not mentioned, default = 10 mins
    oven_needed = 0.0
    microwave_needed = 0.0
    words = []
    for sent in sents:
        words.append(nltk.word_tokenize(sent))

    # add features
    for word in words:
        for i in range(len(word)):
            word[i] = word[i].encode("utf-8")
            word[i] = word[i].lower()
            for key in feas.keys():
                if key != "verbs":
                    if word[i] in corpus[key] and word[i] not in feas[key]:
                        #if word[i] not in feas[key]:
                        feas[key].append(word[i])
                        break
                else:
                    if word[i] in corpus[key]:
                        feas[key] += 1.0
            if word[i] == "oven":
                oven_needed = 1.0
            elif word[i] == "microwave":
                microwave_needed = 1.0
            #if unicode(word[i]).isdigit():
            if i < len(word) - 1:
                if "-" in word[i]:
                    word[i] = word[i].split("-")[-1]
                if word[i].isdigit():
                    if re.match("(minute)s*", word[i+1]):
                    #if word[i+verbs] == "miniutes":
                        time_needed += float(word[i])
                    elif re.match("(second)s*", word[i+1]):
                    #if word[i+verbs] == "seconds":
                        time_needed += float(word[i]) / 60.0
                    elif re.match("(hour)s*", word[i+1]):
                    #if word[i+verbs] == "hours":
                        time_needed += float(word[i]) * 60.0
                    # elif re.match("(day)s*", word[i+1]):
                    # #if word[i+verbs] =="days" or word[i+verbs] == "day":
                    #     time_needed += int(word[i]) * 24 * 60

    if time_needed == 0.0:
        time_needed = 10.0
    for key in feas.keys():
        if key != "verbs":
            ftrs.append(float(len(feas[key])))

    ftrs.append(feas["verbs"])
    ftrs.extend([time_needed, oven_needed, microwave_needed])
    return np.array(ftrs)

# standardization and return mean and std of training data which are used in test data
def standardize(data):
    mean_std = {"mean": [], "std": []}
    for i in range(data.shape[1]):
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        data[:, i] = (data[:, i] - mean) / std
        mean_std["mean"].append(mean)
        mean_std["std"].append(std)
    return mean_std

