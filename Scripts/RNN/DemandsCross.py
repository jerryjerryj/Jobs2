import glob, keras
import numpy as np
from keras.layers import Dense, Activation
from keras.layers import LSTM, SpatialDropout1D,SpatialDropout2D, Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.utils import np_utils
from Scripts.RNN.Preprocess import NgramVocabulary
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def MakeLSTM(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(LSTM(100))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def MakeConvLSTM(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(50, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(Conv1D(filters=512, kernel_size=9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MakeConv(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(Conv1D(filters=512, kernel_size=9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def MakeHiddenSimple(NB_CLASSES):
    N_HIDDEN = 1
    RESHAPED = X_train.shape[1]
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
    model.add(Activation('relu'))
    model.add(Dense(N_HIDDEN))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MakeDataset(pilepathPS, filepathPStest, ngramsSize):
    with open(pilepathPS, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(filepathPStest, 'r', encoding='utf-8') as f2:
        lines.extend(f2.readlines())
    vectors = []
    classes = []
    vocabulary = NgramVocabulary(ngramsSize)
    for line in lines:
        parts = line.split('\t')
        classes.append(int(parts[0]))
        vectors.append(vocabulary.GetIds(parts[1]))

    maxValue = vocabulary.vocabulary.__len__()
    vectors = keras.preprocessing.sequence.pad_sequences(vectors)
    nb_classes = max(classes)+1
    classes = np_utils.to_categorical(classes, nb_classes)
    return vectors, classes, maxValue, nb_classes

pathPS = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
pathPSTest = 'F:\My_Pro\Python\Jobs2\Data\ProfStandartsTest'
testFilesPaths = glob.glob(pathPSTest + "\*.txt")
ngramSize = 5

results = []
for path in testFilesPaths[:1]:
    name = path.split('\\')[-1].split('.')[0]
    vectors, classes, MAX_VALUE, NB_CLASSES = MakeDataset(path,pathPS+'\\'+name+'.txt', ngramSize)
    tosplit = [1]*classes.__len__()
    skf = StratifiedKFold(n_splits=3,shuffle=True)

    for train, test in skf.split(vectors, tosplit):
        model = MakeConvLSTM(MAX_VALUE,NB_CLASSES)
        scores = model.evaluate(vectors[test], classes[test], verbose=0)
        model.fit(vectors[train], classes[train], validation_data=(vectors[test], classes[test]), epochs=2, batch_size=100)
        scores = model.evaluate(vectors[test], classes[test], verbose=0)
        results.append(name + "\t%.2f%%" % (scores[1] * 100))
results = '\n'.join(results)
print(results)