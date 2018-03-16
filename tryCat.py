
from sklearn.preprocessing import MultiLabelBinarizer
from keras.utils import np_utils
classes = [1,2,3,4,1,0]
classes1 = np_utils.to_categorical(classes, 5)
print(classes1)
print(type(classes1))

classesMult = [[1],[2,3],[4,1],[0]]
classes2 = MultiLabelBinarizer().fit_transform(classesMult)
print(classes2)
print(type(classes2))