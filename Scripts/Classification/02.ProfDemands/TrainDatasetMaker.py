from  gensim.models import Word2Vec
import pandas as pd
import pickle
from Scripts.Tools.Averaging import SentenceToVector
from Scripts.W2VLearnings.TokenizeText import TokenizeSentences

profName = '0Администратор БД'
pathToProfessionDemands = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts\\'+profName+'.txt'

with open(pathToProfessionDemands,'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

textClasses = []
currentClass = []
for line in lines:
    splitted = line.split('\t')
    if splitted.__len__()==2:
        currentClass.append(splitted[1])
    elif currentClass.__len__()>0:
        textClasses.append(currentClass)
        currentClass = []
        currentClass.append(splitted[0])
    else:
        currentClass.append(splitted[0])
textClasses.append(currentClass)

wv = Word2Vec.load('F:\My_Pro\Python\Jobs2\Scripts\W2VLearnings\model(add)').wv


classes = []
vectors = []
currentClassId = 0

for textLines in textClasses:
    tokenized = TokenizeSentences(textLines)
    for sentence in tokenized:
        vectors.append(SentenceToVector(wv,sentence))
    for i in range(0,textLines.__len__()):
        classes.append(currentClassId)
    currentClassId+=1

dataset = pd.DataFrame()
dataset['classes'] = classes
dataset['vectors'] = vectors
with open(profName+'.train.dataset','wb') as f:
    pickle.dump(dataset,f)