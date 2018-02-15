from Scripts.Preprocessings.DSCreations.Tools import GetFromPS
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

psPath = 'F:\My_Pro\Python\Jobs2\Data\ProfStandarts'

counts = []
summary = 0
filesPaths = glob.glob(psPath+"\*.txt")
for filePath in filesPaths:
    name = filePath.split('\\')[-1].split('.')[0]
    classesRaw = GetFromPS(filePath)
    classes = []
    [classes.append(c[0]) for c in classesRaw]
    a = np.array(classes)
    unique_elements, counts_elements = np.unique(a, return_counts=True)
    #print(name)
    counts.append(counts_elements.tolist())
    summary+= counts_elements.sum()

print(counts)
result = []
for i in range(0,11):
    line = []
    for c in counts:
        if i<c.__len__():
            line.append(c[i])
        else:
            line.append(0)
    result.append(line)
print(result)


#df = pd.DataFrame(result)
df = pd.DataFrame({'00': result[0], '01': result[1], '02': result[2],
                    '03': result[3], '04': result[4], '05': result[5],
'06': result[6], '07': result[7], '08': result[8],'09': result[9], '10': result[10]
                   })
df.plot(kind='bar', stacked=True,edgecolor='black', cmap='magma')#, cmap='inferno')
plt.xlabel('Total counts is '+str(summary))
plt.show()