import glob, keras
from keras.layers import Dense, Activation
from keras.layers import LSTM, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.utils import np_utils
from Scripts.RNN.Preprocess import NgramVocabulary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

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

def MakeDataset(lines, ngramsSize,nb_classes):
    vectors = []
    classes = []
    vocabulary = NgramVocabulary(ngramsSize)
    for line in lines:
        parts = line.split('\t')
        classesStr = parts[0]
        classes.append([int(s) for s in classesStr.split(';')])
        vectors.append(vocabulary.GetIds(parts[1]))

    maxValue = vocabulary.vocabulary.__len__()
    vectors = keras.preprocessing.sequence.pad_sequences(vectors)
    classes = MultiLabelBinarizer().fit_transform(classes)
    return vectors, classes, maxValue

pathVacancies = 'F:\My_Pro\Python\Jobs2\Data\Vacancies'
filepaths = glob.glob(pathVacancies + "\*.txt")
ngramSize = 3
NB_CLASSES =12

lines =[]
for path in filepaths:
    with open(path, 'r', encoding='utf-8') as f:
        lines.extend(f.readlines())

vectors, classes, MAX_VALUE = MakeDataset(lines, ngramSize,NB_CLASSES)
X_train, X_test, y_train, y_test= train_test_split(vectors,classes,test_size=0.3)

#model = MakeLSTM(MAX_VALUE,NB_CLASSES)
model = MakeHiddenSimple(NB_CLASSES)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=16)

scores = model.evaluate(X_test, y_test, verbose=0)
print("\t%.2f%%" % (scores[1] * 100))
#conv Lstm 729/729 [==============================] - 786s 1s/step - loss: 0.2820 - acc: 0.9119 - val_loss: 0.2650 - val_acc: 0.9113
#	91.13%
#Lstm 91.37%