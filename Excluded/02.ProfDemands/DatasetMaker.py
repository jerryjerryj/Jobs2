import pandas as pd
import glob, gensim
from Scripts.Tools.IO import IOTools
from Scripts.Tools.Averaging import SentenceToVector

PSTokeniedPath = 'F:\My_Pro\Python\Jobs2\Data\TokenizedSentences\ProfStandarts.p'
PSSourcesPath = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
modelPath = 'F:\My_Pro\Python\Jobs2\Scripts\W2VLearnings\model'

tmp = IOTools()

demands = tmp.LoadPickle(PSTokeniedPath)
model = gensim.models.Word2Vec.load(modelPath)
vectors = []
for demand in demands:
    vectors.append(SentenceToVector(model.wv,demand))

dataset = pd.DataFrame()
dataset['demands'] = vectors

classes = []
filesPaths = glob.glob(PSSourcesPath+"\*.txt")
i = 0
for filePath in filesPaths:
    lines = tmp.ReadAllLines(filePath)
    for j in range(0, lines.__len__()):
        classes.append(i)
    i+=1

dataset['classes'] = classes

tmp.DumpPickle(dataset,'02.ProfDemands&Classes.p')
