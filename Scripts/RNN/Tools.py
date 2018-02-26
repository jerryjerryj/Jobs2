import pickle, keras, numpy, glob
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
def MultiToSingle(classes):
    result = []
    for c in classes:
        result.append(c[0])
    return numpy.asarray(result)

def ToNatNumbers(vectors):
    min_num = 0
    for v in vectors:
        if min(v) < min_num:
            min_num = min(v)
    min_num *= -1
    return [[elem + min_num for elem in v] for v in vectors]



def MakeLSTM():
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(100, embedding_vecor_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
