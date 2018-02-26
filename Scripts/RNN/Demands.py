import pickle, keras, numpy, glob
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from Scripts.RNN.Tools import MakeLSTM, ToNatNumbers,MultiToSingle

def ReshapeDataset(datasetPath, makeNaturals):
    dataset = pickle.load(open(datasetPath, 'rb'))
    if makeNaturals:
        X_train = keras.preprocessing.sequence.pad_sequences(ToNatNumbers(dataset['train']['vectors']))
        X_test = keras.preprocessing.sequence.pad_sequences(ToNatNumbers(dataset['test']['vectors']))
    else:
        X_train = keras.preprocessing.sequence.pad_sequences(dataset['train']['vectors'])
        X_test = keras.preprocessing.sequence.pad_sequences(dataset['test']['vectors'])
    y_train = MultiToSingle(dataset['train']['classes'])
    y_test = MultiToSingle(dataset['test']['classes'])
    return X_train, X_test, y_train, y_test


totalReport = ["\nAccuracy:\n"]

datasetsFolder = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
filesPaths = glob.glob(datasetsFolder + "\*.ft")

for datasetPath in filesPaths:
    model = MakeLSTM()
    name = datasetPath.split('\\')[-1].split('.')[0]
    X_train, X_test, y_train, y_test= ReshapeDataset(datasetPath,True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=64)

    scores = model.evaluate(X_test, y_test, verbose=0)
    totalReport.append(name+": %.2f%%" % (scores[1]*100))

report = '\n'.join(totalReport)
print(report)