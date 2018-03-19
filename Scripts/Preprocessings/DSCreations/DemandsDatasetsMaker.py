import pickle, pandas, gensim, glob
from Scripts.Preprocessings.DSCreations.Tools import GetFromPS, GetMulticlassesDemands
from Scripts.Preprocessings.DSCreations.VacanciesDatasetMaker import SentenceToFastTextVector, SentenceToTfIdf,SentenceToAverageWeightedVector,SentenceToAverageTfIdfWeightedVector

demandsTokenizedDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedDemands'
demandsMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
#modelW2VPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\W2V.model'
model2VPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsBooks\\'
ftModelPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\FastText.model'
modelTfIdfDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsDemandsTFIDF'
DSOutputDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
modelTypes =  ['w2vtfidf', 'w2v', 'tfidf', 'ft','d2v']

#
# Делает по всем, но modelType выставлять 3 раза в ручную, ну что за ...нокод
#

def MakeVectors(modelType,vacanciesTokenized,tfIdf, modelW2V,modelD2V, modelFastText):
    vectors = []
    outName = ''
    if modelType == 'w2v':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToAverageWeightedVector(modelW2V.wv, vacancy))
        outName = '.w2v'
    elif modelType == 'tfidf':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToTfIdf(vacancy, tfIdf))
        outName = '.tfidf'
    elif modelType == 'w2vtfidf':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToAverageTfIdfWeightedVector(modelW2V.wv, vacancy, tfIdf))
        outName = '.w2vtfidf'
    elif modelType == 'ft':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToFastTextVector(vacancy, modelFastText))
        outName = '.ft'

    elif modelType == 'd2v':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToAverageWeightedVector(modelD2V.wv, vacancy))
        outName = '.d2v'
    return outName,vectors


def MakeDataset(modelType,tokenized,classes,modelTfIdf,modelW2V,modelD2V, modelFastText):
    df = pandas.DataFrame()
    df['classes'] = classes
    name, vectors = MakeVectors(modelType,tokenized,modelTfIdf,modelW2V, modelD2V,modelFastText)
    df['vectors'] = vectors
    return name, df.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    #w2vModel = gensim.models.Word2Vec.load(modelW2VPath+)

    filesPaths = glob.glob(modelTfIdfDir + "\*.p")

    for tfidfmodelPath in filesPaths:
        for modelType in modelTypes:
            ftModel = gensim.models.FastText.load(ftModelPath)
            name = tfidfmodelPath.split('\\')[-1].split('.')[0]
            tfIdfModel = pickle.load(open(tfidfmodelPath, 'rb'))
            w2vModel = gensim.models.Word2Vec.load(model2VPath + name + '.w2v')
            d2vModel = gensim.models.Doc2Vec.load(model2VPath + name + '.d2v')
            ftModel = gensim.models.FastText.load(model2VPath+ name + '.ft')

            tokenizedTrain =  pickle.load(open(demandsTokenizedDir+'\\'+name+'.p', 'rb'))
            tokenizedTest =  pickle.load(open(demandsTokenizedDir+'\\'+name+'.test.p', 'rb'))

            classesTrain = GetFromPS(demandsMarkedDir+'\\'+name+'.txt')
            classesTest = GetMulticlassesDemands(demandsMarkedDir+'Test\\'+name+'.txt')

            outName, trainDataset = MakeDataset(modelType,tokenizedTrain, classesTrain, tfIdfModel, w2vModel, d2vModel, ftModel)
            outName, testDataset = MakeDataset(modelType,tokenizedTest, classesTest, tfIdfModel, w2vModel,d2vModel, ftModel)
            result = {'test' : testDataset,
                      'train' : trainDataset}
            pickle.dump(result, open(DSOutputDir +'\\'+name+outName, "wb"))




