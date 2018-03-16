from keras.utils import np_utils
import glob, keras
from keras.layers import Dense, Activation
from keras.layers import LSTM, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from Scripts.RNN.Preprocess import NgramVocabulary
from sklearn.model_selection import train_test_split



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
name = 'Специалист по интернет-маркетингу'
#name = 'Администратор БД'
ngramsSize =3

vectors, classes, MAX_VALUE, NB_CLASSES = MakeDataset(pathPS+'\\'+name+'.txt',pathPSTest+'\\'+name+'.txt', ngramsSize)
X_train, X_test, y_train, y_test= train_test_split(vectors,classes,test_size=0.3)

N_HIDDEN = 100
EMBEDDING_VECTOR_LENGTH = 32
BATCH_SIZE = 10
RESHAPED = X_train.shape[1]
model = Sequential()
model.add(Embedding(MAX_VALUE, EMBEDDING_VECTOR_LENGTH))

model.add(SpatialDropout1D(0.2))
model.add(LSTM(50, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(Conv1D(filters=512, kernel_size=9, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(NB_CLASSES, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=16)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy %.2f%%" % (scores[1] * 100))
