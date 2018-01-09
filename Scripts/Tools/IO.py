import pickle

class IOTools:
    #TODO refactor code
    def LoadPickle(self, filepath):
        return pickle.load(open(filepath,'rb'))
    def DumpPickle(self, data, fileName):
        pickle.dump(data, open(fileName, "wb"))
    #мне почему-то кажется, должно быть что-то встроенное
    def ReadAllLines(self, filePath):
        with open(filePath, encoding='utf-8') as f:
            content = f.readlines()
        return content