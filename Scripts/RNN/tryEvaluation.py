import pickle, numpy

def CountAccuracy(seq1, seq2):
    total = seq1.__len__()
    true_predictions = 0
    for i in range(0, total):
        sum = seq1[i]+seq2[i]
        if sum != 1:
            true_predictions+=1
    print(seq1)
    print(seq2)
    print(true_predictions/total)
    return true_predictions, total

def MakeClean(sequence):
    middle = (max(sequence)+min(sequence))/2
    result = []
    for s in sequence:
        if s>=middle:
            result.append(1)
        else:
            result.append(0)
    return numpy.asarray(result)

def MakeClean2(sequence):
    result = []
    for s in sequence:
        if s>=1:
            result.append(1)
        elif 1-s<s and s>0:
            print(s-1)
            print(s)
            result.append(1)
        else:
            result.append(0)
    return numpy.asarray(result)

def MakeClean3(sequence):
    result = []
    maximum = max(sequence)
    minimum = min(sequence)
    for s in sequence:
        if s>=maximum:
            result.append(1)
        elif maximum-s<s and s>minimum:
            print(s-1)
            print(s)
            result.append(1)
        else:
            result.append(0)
    return numpy.asarray(result)

X_test= pickle.load(open('X_test','rb'))
y_test= pickle.load(open('y_test','rb'))
y_predicted= pickle.load(open('y_predicted','rb'))

all_total = 0
all_true = 0
accs = []
for i in range (0, X_test.__len__()):
    true,total = CountAccuracy(y_test[i],MakeClean2(y_predicted[i]))
    print(y_predicted[i])
    all_total+=total
    all_true+=true
    accs.append(true/total)
    print('')

print('All true / all total:')
print(all_true/all_total)

print('Average accuracy:')
print(sum(accs)/accs.__len__())

