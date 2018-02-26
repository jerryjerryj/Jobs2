from gensim.models.fasttext import FastText
from Scripts.Preprocessings.TextModelsCreations.Tools import  CollectSentences, GetProfessionsNames
import pickle


pathSource = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
pathBook = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedBooks\\'
pathOut = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsBooks\\'

booknames = GetProfessionsNames()

for bookname in booknames:
    sentences = CollectSentences(pathSource)
    p = pickle.load(open(pathBook+bookname+'.p', "rb"))
    sentences.extend(p)

    model = FastText(min_count=2)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    model.save(pathOut+bookname+'.ft')