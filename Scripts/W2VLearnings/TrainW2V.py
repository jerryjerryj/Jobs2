import glob, pickle, gensim

def CollectSentences(pathToTokenizedPickles):
    filesPaths = glob.glob(pathToTokenizedPickles+ "\*.p")
    sentences = []
    for filePath in filesPaths:
        p = pickle.load(open(filePath, "rb"))
        sentences.extend(p)
    return sentences

def PrintSourceDataStats(sentences):
    numOfWords = 0
    for s in sentences:
        numOfWords+=s.__len__()
    print('Total number of words in dataset: '+str(numOfWords))

pathSource = 'F:\My_Pro\Python\Jobs2\Data\TokenizedSentences'
sentences = CollectSentences(pathSource)

PrintSourceDataStats(sentences)

model = gensim.models.Word2Vec(sentences, min_count=2, workers=10, iter=100)
model.save('model')