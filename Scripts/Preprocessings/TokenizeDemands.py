import glob, pickle
from Scripts.Preprocessings.Tokenize import TokenizeSentences
if __name__ == "__main__":
    profStandarts = '\ProfStandarts'
    profStandartsTest = '\ProfStandartsTest'

    pathSource = 'F:\My_Pro\Python\Jobs2\Data'
    pathTokenized = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TokenizedDemands'

    # неизменные данные
    filesPaths = glob.glob(pathSource+ profStandarts+"\*.txt")
    for filePath in filesPaths:
        fileName = filePath.split('\\')[-1].split('.')[0]
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = [x.replace('\t',' ') for x in content]
            tokenized = TokenizeSentences(content,True)
            pickle.dump( tokenized, open(pathTokenized+'\\'+fileName+'.p', "wb" ) )

    filesPaths = glob.glob(pathSource + profStandartsTest + "\*.txt")
    for filePath in filesPaths:
        fileName = filePath.split('\\')[-1].split('.')[0]
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            content = [x.replace('\t', ' ') for x in content]
            tokenized = TokenizeSentences(content, True)
            pickle.dump(tokenized, open(pathTokenized + '\\' + fileName + '.test.p', "wb"))
