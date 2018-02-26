import pickle, keras, numpy, glob
from sklearn.model_selection import train_test_split
from Scripts.RNN.Tools import MakeLSTM, ToNatNumbers, MultiToSingle


def ReshapeDataset(datasetPath, makeNaturals):
    dataset = pickle.load(open(datasetPath, 'rb'))
    if makeNaturals:
        X = keras.preprocessing.sequence.pad_sequences(ToNatNumbers(dataset['vectors']))
    else:
        X = keras.preprocessing.sequence.pad_sequences(dataset['vectors'])
    Y = MultiToSingle(dataset['classes'])
    return train_test_split(X,Y,test_size=0.1)

datasetPath= 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets\\w2.dataset'



model = MakeLSTM()
X_train, X_test, y_train, y_test= ReshapeDataset(datasetPath,True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
