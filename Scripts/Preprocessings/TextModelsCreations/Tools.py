import glob, pickle

def CollectSentences(pathToTokenizedPickles):
    filesPaths = glob.glob(pathToTokenizedPickles+ "\*.p")
    sentences = []
    for filePath in filesPaths:
        p = pickle.load(open(filePath, "rb"))
        sentences.extend(p)
    return sentences


def PrintSourceDataStats(name, sentences):
    numOfWords = 0
    for s in sentences:
        numOfWords+=s.__len__()
    print('Total n/o words in '+name+' : '+str(numOfWords))

def GetProfessionsNames():
    pathToPS = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'
    names = []
    filesPaths = glob.glob(pathToPS+ "\*.txt")
    for filePath in filesPaths:
        names.append(filePath.split('\\')[-1].split('.')[0])
    return names
