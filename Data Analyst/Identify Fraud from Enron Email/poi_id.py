#!/usr/bin/python

import pickle
import os
import sys
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features =['salary', 'deferral_payments', 'total_payments',
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# This list contains the features that I am going to use
target_features = ['salary', 'bonus', 'exercised_stock_options',
'long_term_incentive', 'from_this_person_to_poi',
'from_poi_to_this_person', 'shared_receipt_with_poi']

# Ignore poi, salary and exercised stock options. (used only to help some
# operations later on).
help_features = ['poi', 'from_messages', 'to_messages',
'salary', 'exercised_stock_options']

features_list = ['poi'] + target_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Removal of the big outlier
data_dict.pop('TOTAL', None)
my_dataset = data_dict

###############################################################################
# The variable data is a list of lists. Each list represents a person.
###############################################################################
# Indexes:
# 0: POI (0 or 1)
# 1: Salary
# 2: Bonus
# 3: Exercised stock options
# 4: Long term insentive plan
# 5: Number of emails sent from this person to a POI
# 6: Number of emails received by his person from a POI
# 7: Number or emails that this person shared receipt with a POI
###############################################################################

data = featureFormat(my_dataset, features_list, sort_keys = True)

# This variable stores the number of emails sent (index = 1) or received
# a person (index = 2).
help_data = featureFormat(my_dataset, help_features, sort_keys = True)

###############################################################################
### CREATION OF THE MODIFIED DATA VARIABLE CALLED "new_data"
###############################################################################
new_data = range(len(data))
for index in range(len(data)):

    # Some variables to improve readability (I also apply a log transformation
    # to the financial features)
    person = data[index]
    salary = data[index][1]
    bonus = data[index][2]
    stock = data[index][3]
    insentive = data[index][4]
    emails_sent = float(help_data[index][1])
    emails_received = float(help_data[index][2])
    emails_to_poi = data[index][5]
    emails_from_poi = data[index][6]
    emails_shared_with_poi = data[index][7]

    # Earnings (New feature)
    new_data[index] = np.insert(person, 3, salary + bonus)

    # Here I check if a person has sent or received any emails at all. If
    # yes, then I store the proportion of those emails that is connected with 
    # POIs.

    # Proportion of emails sent to POIs
    if emails_sent != 0:
        new_data[index][6] = emails_to_poi/emails_sent
    else:
        new_data[index][6] = 0

    # Proportion of emails received from/shared with POIs
    # (respectively per index)
    if emails_received != 0:
        new_data[index][7] = emails_from_poi/emails_received
        new_data[index][8] = emails_shared_with_poi/emails_received
    else:
        new_data[index][7] = 0
        new_data[index][8] = 0
###############################################################################


###############################################################################
### The new_data variable is again a list of lists.
###############################################################################
### New indexes:
# 0: POI (0 or 1)
# 1: Salary
# 2: Bonus
# 3: Earnings (salary + bonus)
# 4: Exercised stock options
# 5: Long term insentive plan
# 6: Proportion of all emails sent from this person that went to a POI
# 7: Proportion of all emails received by his person that came from a POI
# 8: Proportion or all emails that this person received that shared receipt 
# with a POI
###############################################################################

# Insert the new feature and the rescaled data to the data variable
data = new_data

###############################################################################
### Scatterplot of pairs of data
### UNCOMMENT TO SEE THE PLOT
###############################################################################
# for point in data:
#     poi = int(point[0])
#     salary = point[1]
#     bonus = point[2]
#     earnings = point[3]
#     stock = point[4]
#     insentive = point[5]
#     to_poi = point[6]
#     from_poi = point[7]
#     shared_poi = point[8]
#     colors = ["c", "r"]
#     matplotlib.pyplot.scatter( bonus, stock, color=colors[poi], s=10)
# matplotlib.pyplot.xlabel("bonus")
# matplotlib.pyplot.ylabel("stock")
# matplotlib.pyplot.show()
###############################################################################


# In the following section I used five algorithms to test their performance.
# The gaussian naive Bayes, decision tree, random forest, adaboost,
# k-nearest neighbours

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

# The final_features list contains the names of the features for printing them
final_features = ['salary', 'bonus', 'earnings', 'exercised_stock_options',
 'long_term_incentive', '%_from_this_person_to_poi',
 '%_from_poi_to_this_person', '%_shared_receipt_with_poi']

# Split data into labels and figures
labels, features = targetFeatureSplit(data)

# Collect all classifiers to a list of lists. 
# Indexes for each of tthe lists:
# 0: Name of algorithm
# 1: The respective classifier
classifiers = [['\nNAIVE BAYES \n',
               GaussianNB()],
               ['DECISION TREE \n',
               DecisionTreeClassifier(min_samples_split=2)],
               ['RANDOM FOREST \n',
               RandomForestClassifier(n_estimators=20,
                                      min_samples_split=4)],
               ['ADABOOST \n',
               AdaBoostClassifier(learning_rate=1,
                                  n_estimators=50,
                                  random_state=42)],
               ['K NEAREST NEIGHBOURS \n', 
               KNeighborsClassifier(n_neighbors=1,
                                    weights='uniform',
                                    algorithm='auto',
                                    leaf_size=30,
                                    p=2,
                                    metric='minkowski',
                                    metric_params=None,
                                    n_jobs=1)]]

###############################################################################
### Used to print the feature scores
### UNCOMMENT TO SEE PRINTED VALUES
###############################################################################
# sel = SelectKBest(k='all').fit(features, labels)
# scores = sel.scores_
# print '\nThe feature score of:'
# for index in range(len(final_features)):
#     print '     ', str(index) + ')', '"' + final_features[index] + '"',\
#     'is', scores[index]
# print '\n'
###############################################################################

def fit_and_test(features, labels, clf, my_dataset, features_list):
    ### INPUT ###
    # Takes as input the features, the labels, the classifier, the dataset
    # and the list of features. The first three in order to fit to the
    # classifier and the last two in order to dump them for the tester.py
    # script

    ### RETURNS ###
    # Dumps the my_classifier/my_dataset/my_feature_list pkl files for the
    # tester.py to use them

    # Select the 3 features with the highest scores
    features = SelectKBest(k=3).fit_transform(features, labels)

    # Split the data into a training and a test dataset
    features_train, features_test, \
    labels_train, labels_test = train_test_split(features,
                                                 labels,
                                                 stratify=labels,
                                                 test_size=0.1,
                                                 random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Fit predict and validate results
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print 'tester.py is running to acquire accuracy, precision and recall','\n'
    dump_classifier_and_data(clf, my_dataset, features_list)

# Loop over the classifiers and use tester.py to obtain results for accuracy,
# precision and recall
for classifier in classifiers:
    print classifier[0]
    fit_and_test(features, labels, classifier[1], my_dataset, features_list)
    execfile("tester.py")
