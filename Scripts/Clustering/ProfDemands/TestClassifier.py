import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = pickle.load(open('F:\My_Pro\Python\Jobs2\Scripts\Classification\VacanciesTextsToProfessions\DatasetFromMarkedDemands.p', "rb"))

vectors = df['vectors'].tolist()
classes = df['classes'].tolist()

model = pickle.load(open('model','rb'))

predictions = model.predict(vectors)

f_score = f1_score(predictions, classes, average="micro")
print(f_score)