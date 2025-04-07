# -------------------------------------------------------------------------
# AUTHOR: Sunjay N. Guttikonda
# FILENAME: roc_curve.py
# SPECIFICATION: Uses machine learning (training and test datasets) to determine who cheated on their taxes
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 1.5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
# --> add your Python code here
df = pd.read_csv("cheat_data.csv")
data_training = np.array(df.values)[:, 1:]  # exclude ID

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# --> add your Python code here
X = []
for index, row in df.iterrows():
    try:
        refund = 1 if row['Refund'] == 'Yes' else 0

        marital_status = row['Marital Status']
        if marital_status == 'Single':
            marital = [1, 0, 0]
        elif marital_status == 'Married':
            marital = [0, 1, 0]
        else:  # Divorced
            marital = [0, 0, 1]

        # Convert Taxable Income safely
        income_str = str(row['Taxable Income'])
        if 'k' in income_str:
            income = float(income_str.replace('k', ''))
        else:
            print(f"Unexpected income format: {income_str} — skipping row.")
            continue

        features = [refund] + marital + [income]
        X.append(features)

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue

# X = ?

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
y = [1 if val == 'Yes' else 0 for val in df['Cheat']]

# split into train/test sets using 30% for test
# --> add your Python code here
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3)

# generate random thresholds for a no-skill prediction (random classifier)
# --> add your Python code here
ns_probs = [0 for _ in range(len(testy))]


# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
# --> add your Python code here
dt_probs = dt_probs[:, 1]

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()