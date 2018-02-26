from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import  pickle
from Scripts.Preprocessings.TextModelsCreations.Tools import  CollectSentences, PrintSourceDataStats, GetProfessionsNames

def SentencesToD2VRepresentation(sentences):
    result = []
    for i in range(0, len(sentences)):
        result.append(TaggedDocument(words=sentences[i], tags=['SENT_'+str(i)]))
    return result
    #return [TaggedDocument(words=[u'some', u'words', u'here', u'gfgd', u'aggere'], tags=[u'SENT_1']),
    #        TaggedDocument(words=[u'else', u'words', u'here', u'wffgords', u'hsgfgsere'], tags=[u'SENT_1'])]



pathSource = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
pathBook = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedBooks\\'
pathOut = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsBooks\\'

booknames = GetProfessionsNames()

for bookname in booknames:

    sentences = CollectSentences(pathSource)
    p = pickle.load(open(pathBook+bookname+'.p', "rb"))
    sentences.extend(p)

    PrintSourceDataStats(bookname,sentences)

    sentences = SentencesToD2VRepresentation(sentences)
    model = Doc2Vec(window=20, min_count=2, workers=8, alpha=0.025, min_alpha=0.01, dm=0)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=20)

    model.save(pathOut+bookname+'.d2v')