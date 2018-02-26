import pickle, numpy, pandas

datasetPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets\Администратор БД.w2v'

dataset = pickle.load(open(datasetPath, 'rb'))
train = dataset['train']['vectors']

min_num = 0
for t in train:
    if min(t)<min_num:
        min_num = min(t)
min_num*=-1
natArray2 = [[elem+min_num for elem in t]  for t in train]


print(type(train))
print(pandas.DataFrame (natArray2))