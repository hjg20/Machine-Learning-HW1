import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

X = pd.read_csv('./Data/MADELON/madelon_train.data', 
                         delimiter=' ', 
                         header=None)
X = X.drop(X.columns[-1], axis=1)
y = pd.read_csv('./Data/MADELON/madelon_train.labels', 
                header=None)
Xt = pd.read_csv('./Data/MADELON/madelon_valid.data', 
                         delimiter=' ', 
                         header=None)
Xt = Xt.drop(Xt.columns[-1], axis=1)
yt = pd.read_csv('./Data/MADELON/madelon_valid.labels', 
                header=None)

x = [3, 10, 30, 100, 300]
train_er = []
test_er = []
train_er1 = []
test_er1 = []


for i in x:
    # Using ~sqrt(500) max features
    clf = RandomForestClassifier(n_estimators=i, max_features=22)
    clf.fit(X, y.values.ravel())
    preds = clf.predict(X)
    train_er.append(1 - accuracy_score(y, preds))
    tpreds = clf.predict(Xt)
    test_er.append(1 - accuracy_score(yt, tpreds))
    # Using ~ln(500) max features
    clf1 = RandomForestClassifier(n_estimators=i, max_features=6)
    clf1.fit(X, y.values.ravel())
    preds = clf1.predict(X)
    train_er1.append(1 - accuracy_score(y, preds))
    tpreds = clf1.predict(Xt)
    test_er1.append(1 - accuracy_score(yt, tpreds))

    
# Creating a data frame of all train/test errors with all variations of max features
df = pd.DataFrame()
df['# of Trees'] = x
df['Train Error (sqrt(500))'] = train_er
df['Test Error (sqrt(500))'] = test_er
df['Train Error (ln(500))'] = train_er1
df['Test Error (ln(500))'] = test_er1
print(df)

# Plotting all  errors
plt.plot(x, train_er, x, test_er, x, train_er1, x, test_er1, marker='.')
plt.xlabel('# of Trees')
plt.ylabel('Misclassification Error')
plt.title('Madelon # of Trees in RF vs. Missclassification Error')
train_label = mpatches.Patch(color='blue', label='sqrt(500) train')
test_label = mpatches.Patch(color='orange', label='sqrt(500) test')
train1_label = mpatches.Patch(color='green', label='ln(500) train')
test1_label = mpatches.Patch(color='red', label='ln(500) test')
plt.legend(handles=[train_label, test_label, train1_label, test1_label], loc='upper right')
plt.xticks(x)
plt.show()