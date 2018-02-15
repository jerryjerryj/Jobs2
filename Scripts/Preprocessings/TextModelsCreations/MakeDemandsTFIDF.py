from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, glob

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

def Merged(filesPaths):
    for path in filesPaths:
        psPath = path.split('.')[0] + '.p'
        name = path.split('.')[0].split('\\')[-1]

        tokenized = pickle.load(open(psPath, 'rb'))
        tokenized.extend(pickle.load(open(path, 'rb')))

        TFIDF = MakeModel(tokenized)

        pickle.dump(TFIDF, open(outPath + '\\' + name + '.p', 'wb'))

tokenizedPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedDemands'
outPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsDemandsTFIDF'


filesPaths = glob.glob(tokenizedPath+"\*.test.p")
Merged(filesPaths)


