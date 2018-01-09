import glob
import pickle

import gensim
import pandas

from Scripts.Tools.Averaging import SentenceToVector


def GetClasses(pathToMarked):
    filesPaths = glob.glob(pathToMarked + "\*.txt")
    classes = []
    for filePath in filesPaths:
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for c in content:
                classes.append(int(c.split('\t')[0]))
    return classes

###CHANGE THIS
vacanciesPicklesPath = 'F:\My_Pro\Python\Jobs2\Data\TokenizedSentences\Marked.p'
vacanciesMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\Marked'
###/CHANGE THIS

modelPath = 'F:\My_Pro\Python\Jobs2\Scripts\W2VLearnings\model'


classes = GetClasses(vacanciesMarkedDir)

vacanciesTokenized = pickle.load(open(vacanciesPicklesPath, "rb"))
model = gensim.models.Word2Vec.load(modelPath)

vectors = []
for vacancy in vacanciesTokenized:
    vectors.append(SentenceToVector(model.wv,vacancy))

df = pandas.DataFrame()
df['classes'] = classes
df['vectors'] = vectors

#shuffle
df  = df.sample(frac=1).reset_index(drop=True)
pickle.dump( df, open('DatasetFromMarked.p', "wb" ) )