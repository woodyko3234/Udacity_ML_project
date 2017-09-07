
# coding: utf-8

# ### Step 1: Load the dataset and get understanding about the features

# In[1]:

import pickle
from collections import defaultdict
import numpy as np

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

def counting_machine(dictionary):
    '''Put the original data_dict into this function
        it would count how many nan and false values for each feature.'''
    counter_dict = defaultdict(int)
    for key, value_dict in dictionary.items():
        for feature_key, feature_value in value_dict.items():
            if feature_value == 'NaN' or feature_value == False:
                counter_dict[feature_key] +=1
    return counter_dict


# In[2]:

def ppl_counting(dictionary, value_appointed):
    '''Put the original data_dict into this function and appoint what you want to count 
        ('NaN' or whatever you like)
        it would count how many appointed values for each data point (employee).'''
    counter_dict = defaultdict(int)
    for key, value_dict in dictionary.items():
        counter = 0
        for value in value_dict.values():
            if value == value_appointed:
                counter += 1
            else: continue
        # The codes below is set for checking how poi and NaN value counts distribute
        #if value_dict['poi'] == 0:
        #    counter_dict[key] = counter
        #else:
        #    counter_dict[key] = -counter
        counter_dict[key] = counter
    return counter_dict

NaN_count = ppl_counting(data_dict, 'NaN')


# ### Step 2: Data pre-process and analysis

# In[3]:

def extract_feature_names(dict):
    "Define a function to extract all features in the dict of dict and easier to plot"
    dict_values = list(dict.values())[0]
    dict_values_features = list(dict_values.keys())
    dict_values_features.remove('poi')
    dict_values_features.remove('email_address')
    dict_values_features = sorted(dict_values_features)
    #print(dict_values_features)
    return dict_values_features


# In[4]:

### Store to my_dataset for easy export below.
my_dataset = data_dict

#preprocess on the NaN value in the dictionary!!!
def remove_nan(dict, penalty, remove_outlier = False):
    '''looking through the dataset and 
    turn NaN into -1
    also turn all integers into floats
    and remove email_address'''
    for point, sub_dict in dict.items():
        for key, value in sub_dict.items():
            if key == 'email_address': continue
            if value == 'NaN' or np.isnan(value):
                if key == 'deferred_income':
                    sub_dict[key] = - float(penalty)
                else:
                    sub_dict[key] = float(penalty)
            else:
                sub_dict[key] = float( value )
    if remove_outlier:
        dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
        dict.pop('TOTAL', 0)
        dict.pop('LOCKHART EUGENE E', 0)
    return dict
my_dataset = remove_nan(my_dataset, -10, remove_outlier = True)

def reset_feature_list(my_dataset):
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi']
    features_list.extend(extract_feature_names(my_dataset))
    return features_list
    #Total 20 features including poi as the first feature

features_list = reset_feature_list(my_dataset)


# In[5]:

def is_nan(x):
    return (x is np.nan or x != x)
#create a dict that key is the feature name and values are a list of floats

def dict_with_feature(dataset, sorting=False):
    '''
    define a dict with feature_key to better understand the distribution
    key = feature name (salary, bonus, etc.)
    value = list of values of each person
    '''
    dict_with_features = defaultdict(list)
    for point, sub_dict in dataset.items():
        for key, value in sub_dict.items():
            #email address is a string, so pass to next when facing it
            if key == 'email_address': continue
            #create list to save values
            if value == 'NaN':
                if key == 'deferred_income':
                    value = 10.
                else: 
                    value = -10.
                dict_with_features[key].append( float(value) )
            else:
                dict_with_features[key].append( float(value) )
    #sorting values
    if sorting:
        for key, value in dict_with_features.items():
            dict_with_features[key] = sorted(value, key = lambda x : float('-inf') if is_nan(x) else x)
            #sorting nan reference: 
            #https://stackoverflow.com/questions/4240050/python-sort-function-breaks-in-the-presence-of-nan
    return dict_with_features
    #data points = 146 or 143
    #nan means 0.000000


# In[6]:

dict_with_features = dict_with_feature(my_dataset)
from scipy.stats.stats import pearsonr
for key in dict_with_features.keys():
    print ('PearsonR between poi and ', key, " : ", 
           pearsonr(dict_with_features['poi'], dict_with_features[key]))


# ### Step 3: Adding new features

# In[7]:

#Create new features about ratio of each feature to the mean and median of the feature
mean_dict, median_dict = dict(), dict()
#Create a dict to know each mean of feature
for key, value in dict_with_features.items():
    mean_dict[key] = sum(value)/len(value)
    median_dict[key] = sorted(value)[int(len(value)/2)]


# In[8]:

#Create feature list
for point in my_dataset.keys():
    for key in features_list:
        try: mean_dict[key]
        except: continue
        if key != 'poi' and mean_dict[key] != 0:
            my_dataset[point]['%s_mean_ratio' % key] = my_dataset[point][key] / mean_dict[key]
        if key != 'poi' and median_dict[key] != 0:
            my_dataset[point]['%s_median_ratio' % key] = my_dataset[point][key] / median_dict[key]
#check
#print(my_dataset['KISHKILL JOSEPH G'])


# In[9]:

import matplotlib.pyplot as plt

#Creating new features with messages
for sub_dict in my_dataset.values():
    #Create ratio of messages received from poi to total received messages
    if sub_dict['to_messages'] > 0. or sub_dict['to_messages'] < 0.:
        sub_dict['from_poi_message_ratio'] =         (sub_dict['from_poi_to_this_person']/sub_dict['to_messages']) **2
    else:
        sub_dict['from_poi_message_ratio'] = 0.
    #Create ratio of messages sent to poi to total messages sent
    if sub_dict['from_messages'] > 0. or sub_dict['from_messages'] < 0.:
        sub_dict['to_poi_message_ratio'] =         (sub_dict['from_this_person_to_poi']/sub_dict['from_messages']) **2
    else:
        sub_dict['to_poi_message_ratio'] = 0.

    
#Define a function to run the plotting and set threshold
def plotting_function(dictionary, value1, value2, threshold1, threshold2, threshold_option = False):
    counter_r, counter_b = 0, 0
    for sub_dict in dictionary.values():
        if sub_dict['poi'] > 0.:
            plt.scatter(sub_dict[value1], sub_dict[value2], color = 'r')
            if threshold_option:
                if float(sub_dict[value1]) <= threshold1: print(value1, 'exception: ', sub_dict[value1])
                elif float(sub_dict[value2]) <= threshold2: print(value2, 'exception: ', sub_dict[value2])
                else: counter_r += 1
        else:
            if threshold_option:
                if float(sub_dict[value1]) <= threshold1: continue
                elif float(sub_dict[value2]) <= threshold2: continue
            plt.scatter(sub_dict[value1], sub_dict[value2], color = 'b')
            counter_b += 1
    plt.xlabel(value1)
    plt.ylabel(value2)
    plt.savefig('poi_messages')
    plt.show()
    print(counter_r, counter_b)
#show some figures
plotting_function(my_dataset, 'from_poi_message_ratio', 'to_poi_message_ratio', 0.0004 , 0.03, True )



# ### Step 4: Choosing machine learning algorithm

# In[10]:

def featureFormat(dictionary, features, sort_keys = True):
    '''definitely not a necessary function
        just for the checking function in the tester.py'''
    return_list = []
    if sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()
    for key in keys:
        tmp_list = []
        for feature in features:
            value = dictionary[key][feature]
            tmp_list.append( float(value) )
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        return_list.append( tmp_list )
    return np.array(return_list)

def targetFeatureSplit( data ):
    '''definitely not a necessary function
    just for the checking function in the tester.py'''
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )

    return target, features


# In[11]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
    
data = featureFormat(my_dataset, reset_feature_list(my_dataset), sort_keys = True)
#features order is organized by names of people 

# one data point is consist of zeros n NaN values
labels, features = targetFeatureSplit( data )

#normalize the features before running!
# http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import MaxAbsScaler
min_max_scaler = MaxAbsScaler()
features_normalized = min_max_scaler.fit_transform(features)

#naive_bayes
from sklearn.naive_bayes import GaussianNB
#svm
from sklearn.svm import SVC
#random forest, adaboost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#decision tree
from sklearn.tree import DecisionTreeClassifier
#K nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit


# In[12]:

def algorithm_tester(algorithm, test_times, n_range):
    '''for every PCA n_component in each algorithm
    run pipe.fit 100 times to see how precision and recall scores it is
    test_times can be 100 or 1000
    n_range shall be a list including 2 integers'''
    algorithm_tester = []
    for n in range(n_range[0], n_range[1]):
        clf = Pipeline([('reduce_dim', PCA(n_components = n)), ('clf', algorithm)])
        sss = StratifiedShuffleSplit(test_times, test_size = 0.1, random_state = 42)
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        for train_idx, test_idx in sss.split(features_normalized, labels):
        #reference: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
            features_train = []
            features_test  = []
            labels_train   = []
            labels_test    = []    
            for ii in train_idx:
                features_train.append( features_normalized[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features_normalized[jj] )
                labels_test.append( labels[jj] )
            ### fit the classifier using training set, and test on test set
            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            for prediction, truth in zip(predictions, labels_test):
                if prediction == 0 and truth == 0:
                    true_negatives += 1
                elif prediction == 0 and truth == 1:
                    false_negatives += 1
                elif prediction == 1 and truth == 0:
                    false_positives += 1
                elif prediction == 1 and truth == 1:
                    true_positives += 1
                else:
                    print ("Warning: Found a predicted label not == 0 or 1.")
                    print ("All predictions should take value 0 or 1.")
                    print ("Evaluating performance for processed predictions:")
                    break
        try:
            total_predictions = true_negatives + false_negatives +                                 false_positives + true_positives
            precision = 1.0*true_positives/(true_positives+false_positives)
            recall = 1.0*true_positives/(true_positives+false_negatives)
            accuracy = 1.0*(true_positives + true_negatives)/total_predictions

        except: continue
        if precision >= 0.3 and recall >= 0.3:
            tmp_list = [recall, precision, accuracy] # [avg_recall, avg_precision, avg_accuracy, n_component]
            tmp_list.append(n)
            algorithm_tester.append(tmp_list)
    return algorithm_tester


# In[13]:

best_outcome = algorithm_tester(GaussianNB(), 1000, [9, 10])
print("""Best output in applying PCA method with 58 features included
Pipeline([('reduce_dim', PCA(n_components = 9)), ('clf', GaussianNB())]) \n""", best_outcome)
# [recall score, precision score, accuracy, PCA components]


# In[14]:

#try if we don't add any new features
with open("final_project_dataset.pkl", "rb") as data_file:
    my_dataset_o = pickle.load(data_file)
my_dataset_o = remove_nan(my_dataset_o, 0., remove_outlier = True)
features_list = reset_feature_list(my_dataset_o)
data_o = featureFormat(my_dataset_o, features_list, sort_keys = True)
labels, features = targetFeatureSplit( data_o ) 
features_normalized = min_max_scaler.fit_transform(features)


# In[15]:

best_outcome_o = (algorithm_tester(GaussianNB(),1000, [8, 9]))
print('''Best output in applying PCA method with 19 features included
Pipeline([('reduce_dim', PCA(n_components = 8)), ('clf', GaussianNB())]) \n''',
      best_outcome_o)


# In[16]:

import warnings
#turn warning messages off!
warnings.filterwarnings('ignore')


# In[17]:

def algorithm_tester_tuned(algorithm, tuning_params, test_times):
    '''for every PCA n_component in each algorithm
    run pipe.fit 100 times to see how precision and recall scores it is
    test_times can be 100 or 1000
    n_range shall be a list including 2 integers'''
    algorithm_tester = []
    clf = Pipeline([('minabser', MaxAbsScaler()),
                    ('select', SelectKBest()), 
                    ('clf', algorithm) 
                       ])
    sss = StratifiedShuffleSplit(test_times, test_size = 0.1, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in sss.split(features, labels):
    #reference: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []    
        for ii in train_idx:
            features_train.append( features_normalized[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features_normalized[jj] )
            labels_test.append( labels[jj] )
        ### fit the classifier using training set, and test on test set
        clf.set_params(**tuning_params).fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print ("Warning: Found a predicted label not == 0 or 1.")
                print ("All predictions should take value 0 or 1.")
                print ("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives +                             false_positives + true_positives
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions

    except: return ValueError
    tmp_list = [recall, precision, accuracy] # [avg_recall, avg_precision, avg_accuracy, n_component]
    algorithm_tester.append(tmp_list)
    return algorithm_tester


# In[18]:

algorithms_comparison = dict()
labels, features = targetFeatureSplit( data )
features_normalized = min_max_scaler.fit_transform(features)


# In[19]:

SVC_best_param = {'select__k': 7}
algorithms_comparison['SVC'] = algorithm_tester_tuned(
    SVC(), SVC_best_param, 1000)


# In[20]:

NB_best_param = {'select__k': 16}
algorithms_comparison['Naive Bayes'] = algorithm_tester_tuned(
    GaussianNB(), NB_best_param, 1000)


# In[21]:

DT_best_param = {'select__k': 9}
algorithms_comparison['Decision Tree'] = algorithm_tester_tuned(
    DecisionTreeClassifier(), DT_best_param, 1000)


# In[22]:

RF_best_param = {'select__k': 9}
algorithms_comparison['Random Forest'] = algorithm_tester_tuned(
    RandomForestClassifier(), RF_best_param, 1000)


# In[23]:

adaboost_best_param = {'select__k': 50}
algorithms_comparison['Adaboost'] = algorithm_tester_tuned(
    AdaBoostClassifier(), adaboost_best_param, 1000)


# In[24]:

KNN_best_param = {'select__k': 8}
algorithms_comparison['K Nearest Neighbors'] = algorithm_tester_tuned(
    KNeighborsClassifier(), KNN_best_param, 1000)


# In[25]:

print(algorithms_comparison)


# In[26]:

sss = StratifiedShuffleSplit(n_splits=100, test_size = 0.1, random_state = 42)
param_grid = {'select__k': [8],
             'model__n_neighbors': [3],
             'model__weights': ['distance'], #already tried rbf and sigmoid and worse than poly
             'model__leaf_size': [10]
             } #push up dimention did not help at all
##Search for best_params for KNeighborsClassifier
clf = KNeighborsClassifier()
## Pipeline object
pipe = Pipeline(steps=[('minabser', MaxAbsScaler()),
                    ('select', SelectKBest()), 
                    ('model', clf)   
                    ])
dt_search = GridSearchCV(pipe, param_grid=param_grid, cv=sss,
                    scoring = 'f1')
##print(dt_search.get_params)
dt_search.fit(features, labels)
### Score of best_estimator on the left out data
## Print the optimized parameters used in the model selected from grid search
print('%s' % clf, ' result table: ', dt_search.best_params_)
print('Features index list: ', dt_search.best_estimator_.named_steps['select'].get_support())
knn_bestkfeature = dt_search.best_estimator_.named_steps['select'].get_support()


# In[27]:

KNN_best_param = {'select__k': 8, 'clf__n_neighbors': 3, 'clf__leaf_size': 10,
                 'clf__weights': 'distance'}
algorithms_comparison['K Nearest Neighbors'] = algorithm_tester_tuned(
    KNeighborsClassifier(), KNN_best_param, 1000)
print(algorithms_comparison)


# In[28]:

features_list = reset_feature_list(my_dataset)
print(len(features_list[1:]), len(knn_bestkfeature))
finalized_features_list = []
for boolean, feature in zip(knn_bestkfeature, features_list[1:]):
    if boolean == True and feature not in finalized_features_list:
        finalized_features_list.append(feature)
print(finalized_features_list)


# ### Step 5: Output

# In[29]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import dump_classifier_and_data, test_classifier
clf = Pipeline([('minabser', MaxAbsScaler()),
                    ('reduce_dim', SelectKBest(k = 8)), 
                 ('clf', KNeighborsClassifier(n_neighbors = 3, leaf_size = 10, 
                                              weights = 'distance'))])
features_list = reset_feature_list(my_dataset)
#test final output
test_classifier(clf, my_dataset, features_list, folds = 1000)

# In[30]:

# Finally, output the data
dump_classifier_and_data(clf, my_dataset, features_list)

