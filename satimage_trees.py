# =============================================================================
# Creating and training Decision trees with train/test data from Satimage set
# =============================================================================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

X = pd.read_csv('./Data/satimage/X.dat', 
                         delimiter=' ', 
                         header=None)
y = pd.read_csv('./Data/satimage/Y.dat', 
                header=None)
Xt = pd.read_csv('./Data/satimage/Xtest.dat', 
                         delimiter=' ', 
                         header=None)
yt = pd.read_csv('./Data/satimage/Ytest.dat', 
                header=None)

x = [_+1 for _ in range(12)]
train_er = []
test_er = []

# Creating 12 trees from depth 1 - 12
for i in x: 
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X, y)
    preds = clf.predict(X)
    train_er.append(round(1- accuracy_score(y, preds), 4))
    tpreds = clf.predict(Xt)
    test_er.append(round(1 - accuracy_score(yt, tpreds), 4))

# Creating a Data frame with train/test missclassification error vs. max depth
df = pd.DataFrame()
df['Depth'], df['Train Error'], df['Test Error'] = (x, train_er, test_er)
print(df) #

# Plotting the above data frame
plt.plot(x, train_er, x, test_er, marker='.')
plt.xlabel('Tree Depth')
plt.ylabel('Misclassification Error')
plt.title('Madelon Tree Depth vs. Missclassification Error')
train_label = mpatches.Patch(color='blue', label='Train')
test_label = mpatches.Patch(color='orange', label='Test')
plt.legend(handles=[train_label, test_label], loc='upper right')
plt.xticks(x)
plt.show()

# Printing minimum test error and depth it occured at to console
for i in range(len(x)):
    if min(test_er) == test_er[i]:
        minx = x[i]

print(f'\nMin Test Error occured at depth {minx} with error of '\
      f'{round(min(test_er), 4)}.')
