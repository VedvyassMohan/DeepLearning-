import numpy as np
# Each parent vector is composed recursively from its two child vectors.
emb={"good":np.array([.8,.6]),"bad":np.array([-.8,-.6]),
     "movie":np.array([.2,.1]),"acting":np.array([.3,.2])}
W=np.array([[.7,.2],[.1,.8]])
def compose(left,right): return np.tanh(W@(left+right))
def sentiment(root): return ("Positive" if root.sum()>=0 else "Negative",float(root.sum()))
phrase=compose(emb["movie"],emb["acting"])
for sentence,word in [("good movie acting","good"),("bad movie acting","bad")]:
 root=compose(emb[word],phrase); label,score=sentiment(root)
 print(sentence,"=>",label,"score:",round(score,3))
print("Child-node representations are combined before sentence classification.")
