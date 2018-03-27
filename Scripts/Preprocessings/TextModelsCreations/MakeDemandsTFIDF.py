from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, glob

from keras.preprocessing.text import Tokenizer

def MakeModel(tokenized):
    inPlainTextsFormat = [' '.join(t) for t in tokenized]
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(inPlainTextsFormat)
    idf = vectorizer.idf_
    return dict(zip(vectorizer.get_feature_names(), idf))

def NotMerged(filesPaths):
    for path in filesPaths:
        psPath = path.split('.')[0] + '.p'
        name = path.split('.')[0].split('\\')[-1]

        psTokenized = pickle.load(open(psPath, 'rb'))
        demandsTokenized = pickle.load(open(path, 'rb'))

        psTFIDF = MakeModel(psTokenized)
        demandsTFIDF = MakeModel(demandsTokenized)

        pickle.dump(demandsTFIDF, open(outPath + '\\' + name + '.p', 'wb'))
        pickle.dump(psTFIDF, open(outPath + '\\' + name + '.test.p', 'wb'))

def get_texts_to_matrix(texts, max_features=0):
    tokenizer = Tokenizer(split=" ", lower=True)
    if max_features != 0:
        tokenizer = Tokenizer(split=" ", lower=True, num_words=max_features)

    tokenizer.fit_on_texts(texts)
    matrix_tfidf = tokenizer.texts_to_matrix(texts=texts, mode='tfidf')
    print('Количество текстов:', matrix_tfidf.shape[0])
    print('Количество токенов:', matrix_tfidf.shape[1])
    return matrix_tfidf, tokenizer.word_index

def Merged(filesPaths):
    for path in filesPaths:
        psPath = path.split('.')[0] + '.p'
        name = path.split('.')[0].split('\\')[-1]

        tokenized = pickle.load(open(psPath, 'rb'))
        tokenized.extend(pickle.load(open(path, 'rb')))

        TFIDF, dictionary = get_texts_to_matrix(tokenized)
        result = {'tfidf': TFIDF, 'dictionary': dictionary}

        pickle.dump(result, open(outPath + '\\' + name + '.p', 'wb'))

tokenizedPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedDemands'
outPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsDemandsTFIDF'


filesPaths = glob.glob(tokenizedPath+"\*.test.p")
Merged(filesPaths)


