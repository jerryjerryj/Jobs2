import glob, pickle

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