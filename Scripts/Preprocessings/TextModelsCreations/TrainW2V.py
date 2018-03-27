import gensim
from Scripts.Preprocessings.TextModelsCreations.Tools import  CollectSentences, PrintSourceDataStats

pathSource = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
pathOut = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models'

sentences = CollectSentences(pathSource)

PrintSourceDataStats('',sentences)

model = gensim.models.Word2Vec(sentences, min_count=2, workers=10, iter=100,size=500)
print(model.wv['бд'])
model.save(pathOut+'\\W2V.model')