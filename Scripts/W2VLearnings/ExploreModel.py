import pickle

import gensim

from Scripts.Tools.Averaging import SentenceToVector

vacanciesPath = 'F:\My_Pro\Python\Jobs2\Data\TokenizedSentences\Marked.p'
vacanciesTokenized = pickle.load(open(vacanciesPath, "rb"))

model = gensim.models.Word2Vec.load('model')

vector1 = SentenceToVector(model.wv, vacanciesTokenized[0])
print(vector1)