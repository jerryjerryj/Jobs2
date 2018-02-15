from sklearn.feature_extraction.text import TfidfVectorizer

with open('F:\My_Pro\Python\Jobs2\Data\Vacancies\\2700-2799.txt','r', encoding='utf-8') as f:
    dataset = f.readlines()


vectorizer = TfidfVectorizer()
vectorizer.fit_transform(dataset)
idf = vectorizer.idf_
print (dict(zip(vectorizer.get_feature_names(), idf)))