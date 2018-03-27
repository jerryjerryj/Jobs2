from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, glob

tokenizedPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\Tokenized'
outFile = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\TextModelsCreations\Models\TfIdf.model'


tokenized = []
filesPaths = glob.glob(tokenizedPath+"\*.p")
for path in filesPaths:
    p = pickle.load(open(path,'rb'))
    tokenized.extend(p)

inPlainTextsFormat = [' '.join(t) for t in tokenized]


vectorizer = TfidfVectorizer()
vectorizer.fit_transform(inPlainTextsFormat)
idf = vectorizer.idf_
dictionary = dict(zip(vectorizer.get_feature_names(), idf))

pickle.dump(dictionary, open(outFile,'wb'))