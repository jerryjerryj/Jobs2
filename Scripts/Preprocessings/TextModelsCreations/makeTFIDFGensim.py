import gensim.downloader as api
import glob, pickle
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

tokenizedPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized\Vacancies.p'
outFile = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\TfIdf.model'


dataset = pickle.load(open(tokenizedPath,'rb'))

dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert dataset to BoW format
model = TfidfModel(corpus)  # fit model
result = []
for  i in range(0, dataset.__len__()):
    doc_tfidf_dict = {}
    vector = model[corpus[i]]
    for v in vector:
        doc_tfidf_dict[dct[v[0]]]=v[1]
    result.append(doc_tfidf_dict)

pickle.dump(result,open(outFile,'wb'))

