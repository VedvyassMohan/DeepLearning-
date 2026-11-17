import math

labels = ['Low']*6 + ['High']*4
counts = {label: labels.count(label) for label in set(labels)}
n = len(labels)
entropy = -sum((c/n)*math.log2(c/n) for c in counts.values())
print('Class counts:', counts)
print('Entropy:', round(entropy, 4), 'bits')
