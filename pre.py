import os
import nltk
from gensim.models import Word2Vec
import sklearn
import numpy as np
import argparse
import cross_validation
import re
# import prettytable as pt
# from sklearn.linear_model import LogisticRegression
# from keras.models import Sequential
# from keras.layers import Input, Dense, Flatten,LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding



global MAX_NUM_WORDS
MAX_NUM_WORDS = 5000
global MAX_SEQUENCE_LENGTH
MAX_SEQUENCE_LENGTH = 100
global DATA_NUM
DATA_NUM = 1047
global DIM_EMBED
DIM_EMBED = 50
global VALIDATION_SPLIT
VALIDATION_SPLIT = 0.2

def preprocess_corpus():
    #corpus
    path = "lexicon/"
    files = os.listdir(path)
    global corpus
    corpus = {}
    for file in files:
        f = open(path + file, "r")
        corpus[file] = f.read().splitlines()

    #training data recipe
    global Data, Label
    Data = []
    Label = []
    f1 = open("data/data.1047")
    f2 = open("data/label.1047")
    for i in range(DATA_NUM):
        line = f1.readline().splitlines()[0]
        label = f2.readline().splitlines()[0]
        Data.append(line)
        Label.append(int(label))
    f1.close()
    f2.close()
    # trans = {"0":"easy", "1":"medium", "2":"hard"}
    # original_Label = []
    # for label in Label:
    #     original_Label.append(trans[str(label)])
    original_Label = Label
    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(Label) + 1))
    Label = label_binarizer.transform(Label)
    return corpus, Data, Label, original_Label
    # for i in range(DATA_NUM):
    #     line = f1.readline().splitlines()[0]
    #     lab = f2.readline().splitlines()[0]
    #     name, recipe = line.split("Recipe:")
    #     name = name.split("Name:")[1].strip().strip("\"")
    #     Recipe[name] = recipe
    #     Label[name] = int(lab)

def word2vec(textdata, Size=100):
    text = []
    for sents in textdata:
        line = sents.decode('utf-8')
        tokens = nltk.word_tokenize(line)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        text.append(tokens)
    model = Word2Vec(text, size=Size, window=5, min_count=1, workers=4)
    model.train(text, total_examples=len(text), epochs=10)
    model.save("data/word2vec")
    model.wv.save_word2vec_format('data/word2vec.txt', binary=False)
    return text

def extrac_vec(text, num_word = 100, Dim=100):
    model = Word2Vec.load("data/word2vec")
    data_vec = []
    for row in text:
        vec = []
        count = 0
        for word in row:
            if count < num_word:
                vec.append(model.wv[word])
                count +=1
            else:
                break
        if count < num_word:
            for i in range(num_word - count):
                vec.append(np.zeros(Dim))
        data_vec.append(np.array(vec))
    data_vec = np.array(data_vec)
    return data_vec

def get_features(data):
    training_data = []
    for sentences in data:
        sentences = sentences.decode('utf-8')
        ftrs = []
        sents = nltk.sent_tokenize(sentences)
        ftrs.append(len(sents))
        feas = {}
        for key in corpus.keys():
            if key != "verbs":
                feas[key] = []
            else:
                feas[key] = 0.0
        time_needed = 0.0    # unit miniute, if not mentioned, default = 10 mins
        # oven_needed = 0.0
        # microwave_needed = 0.0
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
                            feas[key].append(word[i])
                            break
                    else:
                        if word[i] in corpus[key]:
                            feas[key] += 1.0
                # if word[i] == "oven":
                #     oven_needed = 1.0
                # elif word[i] == "microwave":
                #     microwave_needed = 1.0
                #if unicode(word[i]).isdigit():
                if i < len(word) - 1:
                    if "-" in word[i]:
                        word[i] = word[i].split("-")[-1]
                    if word[i].isdigit():
                        if re.match("(minute)s*", word[i+1]):
                            time_needed += float(word[i])
                        elif re.match("(second)s*", word[i+1]):
                            time_needed += float(word[i]) / 60.0
                        elif re.match("(hour)s*", word[i+1]):
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
        ftrs.apeend(time_needed)
        training_data.append(np.array(ftrs))
    global colnames
    colnames = ["sent_length"]
    for key in feas.keys():
        if key != "verbs":
            colnames.append(key)
    colnames.extend(["verbs", "time_needed"])
    return np.array(training_data)

def split_data(data, label, VALIDATION_SPLIT=0.2):
    data = np.array(data)
    indices = np.arange(data.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = label[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = label[-nb_validation_samples:]
    return x_train,y_train,x_val,y_val

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recipe Difficulty Level Judgement',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--classifier", "-clf", default="LR", choices=["LR", "LSTM"], help="which classifier to ues")
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    corpus, Data, Label, original_Label = preprocess_corpus()

    if args.classifier =='LR':
        x_train, y_train, x_val, y_val =split_data(Data, original_Label,)
        training_data = get_features(x_train)

        cverror = []
        C = [0.001, 0.01, 0.1, 1, 10, 100]
        for c in C:
            lr = LogisticRegression(penalty = "l2", C = c,  multi_class = "ovr")
            cverror.append(cross_validation.cv_error(lr, dataset=training_data, datalabel=y_train, cv=5))
        best_c = C[cverror.index(min(cverror))]
        lr = LogisticRegression(penalty="l2", C=best_c,  multi_class="ovr")

        mean_std = standardize(training_data)
        #print training_data
        lr.fit(training_data, y_train)
        trainscore = lr.score(training_data, y_train)
        # standardize test data
        testdata = []
        testdata = get_features(x_val)
        for i in range(testdata.shape[1]):
            testdata[:, i] = (testdata[:, i] - mean_std["mean"][i]) / mean_std["std"][i]
        score = lr.score(testdata, y_val)
        tb = pt.PrettyTable()
        tb.field_names = ["1/Lambda", "cross_validation_accuracy"]
        for i in range(len(C)):
            tb.add_row([C[i],1-cverror[i]])
        print tb
        print "minimum cv accuracy:", 1-min(cverror)
        print "best_constant:", best_c
        print "training accuracy:", trainscore
        print "test accuracy:", score

    elif args.classifier == 'LSTM':
        C2 = [10,20,30,50,75,100,150,200]
        Test_acc = []
        Train_acc = []
        
        for c in C2:
            data = word2vec(Data,Size=c)
            data_vec = extrac_vec(data, Dim=c)
            x_train, y_train, x_val, y_val =split_data(data_vec, label = Label,)

            model = Sequential()
            model.add(LSTM(100, input_shape=(100, c), dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(64))
            model.add(Activation('tanh'))
            model.add(Dense(Label.shape[1], activation='softmax'))
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            history = model.fit(x_train, y_train,
                                validation_data=(x_val,y_val),
                                batch_size=8,#8
                                epochs=10,
                                verbose=1,
                                validation_split=0.1)
            Train_acc.append(model.evaluate(x_train, y_train)[1])
            Test_acc.append(model.evaluate(x_val, y_val)[1])
        tb2 = pt.PrettyTable()
        tb2.field_names = ['WordVec_Dimension', 'Training_Accuracy','Test_Accuracy']
        for i in range(len(C2)):
            tb2.add_row([C2[i],Train_acc[i], Test_acc[i]])
        print tb2





