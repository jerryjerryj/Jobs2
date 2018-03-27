import gensim, pickle
from Scripts.Preprocessings.TextModelsCreations.Tools import  CollectSentences, PrintSourceDataStats, GetProfessionsNames

pathSource = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
pathBook = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedBooks\\'
pathOut = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsBooks\\'

booknames = GetProfessionsNames()

for bookname in booknames:
    sentences = CollectSentences(pathSource)
    p = pickle.load(open(pathBook+bookname+'.p', "rb"))
    sentences.extend(p)

    PrintSourceDataStats(bookname,sentences)

    model = gensim.models.Word2Vec(sentences, min_count=2, workers=10, iter=100,size=500)
    model.save(pathOut+bookname+'.w2v')