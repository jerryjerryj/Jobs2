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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

def CountF1Measure(y_pred,y_true):
    y_pred_classes =np.argmax(y_pred, axis=1).tolist()
    y_true_classes = [y.tolist().index(1) for y in y_true]

    return f1_score(y_true_classes, y_pred_classes, average='macro')#, accuracy_score(y_true_classes,y_pred_classes)

def MakeLSTM(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(LSTM(100))
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def MakeConvLSTM(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(50, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(Conv1D(filters=512, kernel_size=9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(NB_CLASSES, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def MakeConv(MAX_VALUE,NB_CLASSES):
    EMBEDDING_VECTOR_LENGTH = 32
    model = Sequential()
    model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))
    model.add(Conv1D(filters=512, kernel_size=9, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(NB_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        classesStr = parts[0]
        classes.append([int(classesStr.split(';')[0])])
        vectors.append(vocabulary.GetIds(parts[1]))

    maxValue = vocabulary.vocabulary.__len__()
    vectors = keras.preprocessing.sequence.pad_sequences(vectors)
    nb_classes = max(classes)[0]+1
    classes = np_utils.to_categorical(classes, nb_classes)
    return vectors, classes, maxValue, nb_classes



pathPS = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
pathPSTest = 'F:\My_Pro\Python\Jobs2\Data\ProfStandartsTest'
testFilesPaths = glob.glob(pathPSTest + "\*.txt")
ngramSize = 2

results = []
for path in testFilesPaths:
    name = path.split('\\')[-1].split('.')[0]
    vectors, classes, MAX_VALUE, NB_CLASSES = MakeDataset(path,pathPS+'\\'+name+'.txt', ngramSize)
    tosplit = [1]*classes.__len__()
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    av = []
    f1 = []
    for train, test in skf.split(vectors, tosplit):
        model = MakeConvLSTM(MAX_VALUE,NB_CLASSES)
        scores = model.evaluate(vectors[test], classes[test], verbose=0)
        model.fit(vectors[train], classes[train], validation_data=(vectors[test], classes[test]), epochs=40, batch_size=100)
        scores = model.evaluate(vectors[test], classes[test], verbose=0)
        y_pred = model.predict(vectors[test])
        f1_macro= CountF1Measure(y_pred,classes[test])
        results.append(name + "\tacc - %.4f" % (scores[1])+ "\tf1 - %.4f" % (f1_macro))
        av.append(scores[1])
        f1.append(f1_macro)
    results.append('%.4f'%(sum(av)/av.__len__())+'\t'+'%.4f'%(sum(f1)/f1.__len__()))
results = '\n'.join(results)
print(results)