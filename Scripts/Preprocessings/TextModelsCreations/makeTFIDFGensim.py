
import gensim.downloader as api
import glob, pickle, pandas
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from keras.preprocessing.text import Tokenizer
def get_texts_to_matrix(texts, max_features=0):
    tokenizer = Tokenizer(split=" ", lower=True)
    if max_features != 0:
        tokenizer = Tokenizer(split=" ", lower=True, num_words=max_features)

    tokenizer.fit_on_texts(texts)
    matrix_tfidf = tokenizer.texts_to_matrix(texts=texts, mode='tfidf')
    print('Количество текстов:', matrix_tfidf.shape[0])
    print('Количество токенов:', matrix_tfidf.shape[1])
    return matrix_tfidf, tokenizer.word_index
def GetExcluded(dictionary, corpus,min_counts):
    counts_by_id = {}
    for corp in corpus:
        for c in corp:
            id = c[0]
            counter = c[1]
            if id in counts_by_id.keys():
                counts_by_id[id]+=counter
            else:
                counts_by_id[id] = counter
    exclude = []
    for key in counts_by_id.keys():
        if counts_by_id[key]<min_counts:
            exclude.append(dictionary[key])
    return exclude

def MakeDict():
    result = []
    # exclude = GetExcluded(dct,corpus,50)
    # print (exclude.__len__())
    for i in range(0, dataset.__len__()):
        doc_tfidf_dict = {}
        vector = model[corpus[i]]
        for v in vector:
            # if dct[v[0]] not in exclude:
            doc_tfidf_dict[dct[v[0]]] = v[1]
        result.append(doc_tfidf_dict)

    return result

tokenizedPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized\Vacancies.p'
outFile = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\TfIdf.model'


dataset = pickle.load(open(tokenizedPath,'rb'))
#dataset =[['one','two','three'],['two','three'],['one','kjdghsljh']]

dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert dataset to BoW format
model = TfidfModel(corpus)  # fit model

tfidf, dictionary = get_texts_to_matrix(dataset)
result = {'tfidf': tfidf, 'dictionary':dictionary}
#print(result)
pickle.dump(result,open(outFile,'wb'))


'''for  i in range(0, dataset.__len__()):
    vectors = model[corpus[i]]
    for x in vectors:
        print(x)'''

# сделать TF-IDF по коду юлия