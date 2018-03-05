class NgramVocabulary:
    vocabulary = [' ']
    nextId = 1
    def __init__(self, n):
        self.n = n

    def SplitNgramsLetters(self, sentence):
        sentence = sentence.lower()
        cleaned = []
        isSpace = False
        for symbol in sentence:
            if (symbol >= 'а' and symbol <= 'я') or (symbol >= 'a' and symbol <= 'z'):
                isSpace = False
                cleaned.append(symbol)
            elif not isSpace:
                isSpace = True
                cleaned.append(' ')
        sentence = ''.join(cleaned)
        result = []
        for i in range(0, sentence.__len__() - self.n+1):
            result.append(sentence[i:i + self.n])
        return result

    def GetIds(self, sentence):
        result = []
        ngrams = self.SplitNgramsLetters(sentence)
        for ngram in ngrams:
            if ngram in self.vocabulary:
                result.append(self.vocabulary.index(ngram))
            else:
                self.vocabulary.append(ngram)
                result.append(self.nextId)
                self.nextId+=1
        return result




'''ngrams = 3
#sen = 'Предоставление и контроль прав доступа пользо&&% 43 23 вателей к БД: '
sen1 = 'Раз и под'
sen2 = 'под и раз'


ngramVocabulary = NgramVocabulary(ngrams)
sen2numbers = ngramVocabulary.GetIds(sen1)
print(sen2numbers)
sen2numbers = ngramVocabulary.GetIds(sen2)
print(sen2numbers)
print(ngramVocabulary.vocabulary)'''