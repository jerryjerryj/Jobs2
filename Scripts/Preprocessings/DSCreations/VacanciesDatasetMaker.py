import pickle, pandas, gensim
from Scripts.Preprocessings.DSCreations.Tools import GetMulticlasses
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#####   CHANGEABLE PARAMS
vacanciesPicklesPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized\Vacancies.p'
vacanciesNsPicklesPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized\\NS\Vacancies.ns.p'
vacanciesMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\Vacancies'
modelPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\W2V.model'
ftModelPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\FastText.model'
modelTfIdfPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\TfIdf.model'
DSOutputPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets'
modelTypes = ['tfidf']#'w2vtfidf', 'ft','d2v','w2v']
#####


def SentenceToAverageWeightedVector(wv, sentence):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word in wv.vocab:
                vectors[index] = wv[word]
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector

def SentenceToAverageTfIdfWeightedVector(wv, sentence, tfidf):
    vectors = pandas.DataFrame()
    index = 0
    try:
        for word in sentence:
            if word not in tfidf.keys():
                tf_idf = 0
            else:
                tf_idf = tfidf[word]
            if word in wv.vocab:
                vectors[index] = wv[word]*tf_idf
            index += 1
        vectors = vectors.transpose()
        vector = vectors.mean().values.tolist()
    except Exception:
        return []
    return vector

def SentenceToTfIdf(sentence, tfidf, keys):
    #keys = list(tfidf.keys())
    vector = [0] * keys.__len__()
    for word in sentence:
        if word in keys:
            index = keys.index(word)
            temp = tfidf[word]
            vector[index] = tfidf[word]
    return vector
    #return [sum(vector) / float(len(vector))]

def ToTfIdfReduced(tfidfs):
    maxLen = 0
    for t in tfidfs:
        length = t.keys().__len__()
        if maxLen<length:
            maxLen = length
    vectors = []
    for tfidf in tfidfs:
        index = 0
        vector = [0] * maxLen
        for key, value in tfidf.items():
            vector[index] = value
            index+=1
        vectors.append(vector)
    return vectors

def SentenceToFastTextVector(sentence,ft):
    vectors = pandas.DataFrame()
    index = 0
    for word in sentence:
        vectors[index] = ft[word]
        index += 1
    vectors = vectors.transpose()
    vector = vectors.mean().values.tolist()
    return vector

'''def SentenceToTfIdf(sentences):
    spaced = [' '.join(s) for s in sentences]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(spaced)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, [0,0,0,1,1])
    return X_train_tfidf'''

def GetTFIDFKeys(tfidfs):
    keys = []
    for tfidf in tfidfs:
        keys.extend(tfidf.keys())
    return keys

if __name__ == "__main__":
    vacanciesTokenized = pickle.load(open(vacanciesPicklesPath, "rb"))
    vacanciesTokenizedNS = pickle.load(open(vacanciesNsPicklesPath, "rb"))
    model = gensim.models.Word2Vec.load(modelPath)
    ftModel = gensim.models.FastText.load(ftModelPath)
    tfIdf= pickle.load(open(modelTfIdfPath, 'rb'))
    tfIdfKeys = GetTFIDFKeys(tfIdf)

    classes = GetMulticlasses(vacanciesMarkedDir)


    for modelType in modelTypes:
        vectors = []
        outName = ''
        if modelType == 'w2v':
            for vacancy in vacanciesTokenized:
                vectors.append(SentenceToAverageWeightedVector(model.wv, vacancy))
            outName = '\\W2V.dataset'
        elif modelType == 'tfidf':
            vectors = ToTfIdfReduced(tfIdf)
            '''index = 0
            #vectors = list( SentenceToTfIdf(vacanciesTokenized))
            for vacancy in vacanciesTokenized:
                vectors.append(SentenceToTfIdf(vacancy,tfIdf[index],tfIdfKeys))
                index+=1'''
            outName = '\\TfIdf.dataset'
        elif modelType == 'w2vtfidf':
            index = 0
            for vacancy in vacanciesTokenized:
                vectors.append(SentenceToAverageTfIdfWeightedVector(model.wv, vacancy,tfIdf[index]))
                index+=1
            outName = '\\W2VTfIdf.dataset'
        elif modelType == 'ft':
            for vacancy in vacanciesTokenizedNS:
                vectors.append(SentenceToFastTextVector(vacancy,ftModel))
            outName = '\\FT.dataset'

        df = pandas.DataFrame()
        df['classes'] = classes
        df['vectors'] = vectors

        df = df.sample(frac=1).reset_index(drop=True)

        pickle.dump(df, open(DSOutputPath+outName, "wb" ))

