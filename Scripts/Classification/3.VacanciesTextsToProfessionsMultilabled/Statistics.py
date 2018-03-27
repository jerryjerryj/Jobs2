import pickle
import numpy as np
import matplotlib.pyplot as plt
from Scripts.Preprocessings.DSCreations.Tools import GetMulticlasses

'''modelPath = 'F:\My_Pro\Python\Jobs2\Scripts\Preprocessings\DSCreations\DataSets\W2V.dataset'
dataset = pickle.load(open(modelPath,'rb'))
classes = dataset['classes'].tolist()'''

vacanciesMarkedDir = 'F:\My_Pro\Python\Jobs2\Data\Vacancies'
classes = GetMulticlasses(vacanciesMarkedDir)
print(classes.__len__())

all = []
[all.extend(c) for c in classes]

'''remove = [2]
removed_all =[]
for a in all:
    if a not in remove:
        removed_all.append(a)
all = removed_all'''

a = np.array(all)
unique_elements, counts_elements = np.unique(a, return_counts=True)
unique_elements = [u+1 for u in unique_elements]
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

plt.bar(unique_elements,counts_elements)
plt.yticks(range(0,100,4))
plt.xticks(range(0,13,1))
plt.xlabel('Вакансия')
plt.ylabel('Количество входящих текстов')
plt.show()