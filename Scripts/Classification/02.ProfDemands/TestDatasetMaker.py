import pickle

import pandas as pd
from  gensim.models import Word2Vec

from Scripts.Preprocessings.Tokenize import TokenizeSentences
from Scripts.Tools.Averaging import SentenceToVector

__SUPPORT_MISFIT_DEMAND__ = False

#
#
# в разметке ОБЯЗАННОСТИ
#
#


profName = '0Администратор БД'
pathToProfessionDemands = 'F:\My_Pro\Python\Jobs2\Data\ProfStandartsTest\\'+profName+'.txt'


with open(pathToProfessionDemands,'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

classes = []
texts = []
for line in lines:
    splitted = line.split('\t')
    cl = int(splitted[0])
    if cl is not -1 or __SUPPORT_MISFIT_DEMAND__:
        classes.append(cl)
        texts.append(splitted[1])

wv = Word2Vec.load('F:\My_Pro\Python\Jobs2\Scripts\W2VLearnings\model(add)').wv
vectors = []
tokenized = TokenizeSentences(texts)
for sentence in tokenized:
    vectors.append(SentenceToVector(wv,sentence))

dataset = pd.DataFrame()
dataset['classes'] = classes
dataset['vectors'] = vectors
with open(profName+'.test.dataset','wb') as f:
    pickle.dump(dataset,f)

print(dataset)