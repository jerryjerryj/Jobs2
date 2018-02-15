import pickle, pandas, gensim, glob
from Scripts.Preprocessings.DSCreations.Tools import GetFromPS, GetMulticlassesDemands
from Scripts.Preprocessings.DSCreations.VacanciesDatasetMaker import SentenceToTfIdf,SentenceToAverageWeightedVector,SentenceToAverageTfIdfWeightedVector

demandsTokenizedDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedDemands'
demandsMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
modelW2VPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\W2V.model'
modelTfIdfDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\ModelsDemandsTFIDF'
DSOutputDir = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DemandsDatasets'
modelType = 'w2vtfidf' # w2v tfidf w2vtfidf fasttext

def MakeVectors(vacanciesTokenized,tfIdf, modelW2V):
    vectors = []
    outName = ''
    if modelType == 'w2v':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToAverageWeightedVector(modelW2V.wv, vacancy))
        outName = 'W2V.dataset'
    elif modelType == 'tfidf':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToTfIdf(vacancy, tfIdf))
        outName = 'TfIdf.dataset'
    elif modelType == 'w2vtfidf':
        for vacancy in vacanciesTokenized:
            vectors.append(SentenceToAverageTfIdfWeightedVector(modelW2V.wv, vacancy, tfIdf))
        outName = 'W2VTfIdf.dataset'
    return outName,vectors


def MakeDataset(tokenized,classes,modelTfIdf,modelW2V):
    df = pandas.DataFrame()
    df['classes'] = classes
    name, vectors = MakeVectors(tokenized,modelTfIdf,modelW2V)
    df['vectors'] = vectors
    return name, df.sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":
    model = gensim.models.Word2Vec.load(modelW2VPath)

    filesPaths = glob.glob(modelTfIdfDir + "\*.p")

    for tfidfmodelPath in filesPaths:
        name = tfidfmodelPath.split('\\')[-1].split('.')[0]
        tfIdfModel = pickle.load(open(tfidfmodelPath, 'rb'))

        tokenizedTrain =  pickle.load(open(demandsTokenizedDir+'\\'+name+'.p', 'rb'))
        tokenizedTest =  pickle.load(open(demandsTokenizedDir+'\\'+name+'.test.p', 'rb'))

        classesTrain = GetFromPS(demandsMarkedDir+'\\'+name+'.txt')
        classesTest = GetMulticlassesDemands(demandsMarkedDir+'Test\\'+name+'.txt')

        outName, trainDataset = MakeDataset(tokenizedTrain,classesTrain,tfIdfModel,model)
        outName, testDataset = MakeDataset(tokenizedTest,classesTest,tfIdfModel,model)
        result = {'test' : testDataset,
                  'train' : trainDataset}
        pickle.dump(result, open(DSOutputDir +'\\'+name+outName, "wb"))





