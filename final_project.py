
# coding: utf-8

# #Identify Fraud from Enron Email - Intro to ML Project
# ##Woody Yao
# 
# ##Introduction 
# In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. The scandal has brought great impacts on accounting worldwide, which I majored in university, making me more interested in digging into the dataset. In this project, I would play a detective, and put my new skills into it by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. 
# Please note that the python version I dealt with was 3.5, which was different from the version the starter codes applied in many ways. As a result, I would modify the starter codes to keep them out of malfunctioning.

# ### Step 1: Load the dataset and get understanding about the features

# In[55]:

import pickle
from collections import defaultdict
import numpy as np

### Load the dictionary containing the dataset
with open("/Users/KunWuYao/GitHub/Udacity_ML_projects/final_project/final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

print ("Data type: ", type(data_dict))
print ("Data frame and data points in total: ", list(data_dict.items())[0:1], len(data_dict.items()))
print ("Total features number: ", len(list(data_dict.items())[0][1]))

def counting_machine(dictionary):
    '''Put the original data_dict into this function
        it would count how many nan and false values for each feature.'''
    counter_dict = defaultdict(int)
    for key, value_dict in dictionary.items():
        for feature_key, feature_value in value_dict.items():
            if feature_value == 'NaN' or feature_value == False:
                counter_dict[feature_key] +=1
    return counter_dict

print(counting_machine(data_dict))


# ##Data Structure
# After loading the dataset and printing out the first data point, we can see the data structure is a dictionary with 146 data point in total. The dictionary key was the person's name, and the value was another dictionary with 21 key-value pairs in total, which contained the names of all the features and their values for that person. The features in the data fell into three major types, namely financial features, email features and POI labels. 
# See the first data point with dict.items method for example:
# 
# [('PRENTICE JAMES', {'to_messages': 'NaN', 'director_fees': 'NaN', 'restricted_stock': 208809, 'bonus': 'NaN', 'deferred_income': 'NaN', 'email_address': 'james.prentice@enron.com', 'long_term_incentive': 'NaN', 'other': 'NaN', 'loan_advances': 'NaN', 'poi': False, 'salary': 'NaN', 'restricted_stock_deferred': 'NaN', 'deferral_payments': 564348, 'from_this_person_to_poi': 'NaN', 'total_stock_value': 1095040, 'from_poi_to_this_person': 'NaN', 'exercised_stock_options': 886231, 'from_messages': 'NaN', 'total_payments': 564348, 'shared_receipt_with_poi': 'NaN', 'expenses': 'NaN'})]
# 
# ###financial features: 
# ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)
# 
# ###email features: 
# ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
# 
# ###POI label: 
# [‘poi’] (boolean, represented as integer)
# 
# The interesting and hard part of the dataset was that the distribution of the non-POI's to POI's was very skewed, given that from the 146 there were only 18 people or data points labeled as POI's. I was interested in labeling every person in the dataset into either a POI or a non-POI (POI stands for Person Of Interest).
# 

# ###Features without value
# By digging the data structure more deeply, I found that not only POI distribution was very skewed, but there were a lot of none values expressed by string 'NaN'. Find the dictionary addressing how many NaN or false in each feature below:
# {'other': 53, 'deferred_income': 97, 'bonus': 64, 'restricted_stock': 36, 'deferral_payments': 107, 'director_fees': 129, 'email_address': 35, 'to_messages': 60, 'restricted_stock_deferred': 128, 'from_messages': 60, 'salary': 51, 'long_term_incentive': 80, 'total_payments': 21, 'poi': 128, 'from_this_person_to_poi': 80, 'shared_receipt_with_poi': 60, 'total_stock_value': 20, 'exercised_stock_options': 44, 'expenses': 51, 'loan_advances': 142, 'from_poi_to_this_person': 72}
# 
# From the dictionary I found that even the salary, total payments, and messages sent and received got many none values, which might make the POI prediction more difficult.

# In[56]:

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
print(sorted(NaN_count.items(), key = lambda counting: counting[1],
                      reverse = True))


# ###None values counting for each person
# After knowing that there were a lot of none values in each feature, I created another dictionary to count the amount of missed values for each person. Here's the persons and their missed values in total (only including data points with count higher than 15):
# 
# [('LOCKHART EUGENE E', 20), ('WHALEY DAVID A', 18), ('WROBEL BRUCE', 18), ('GRAMM WENDY L', 18), ('THE TRAVEL AGENCY IN THE PARK', 18), ('CLINE KENNETH W', 17), ('SAVAGE FRANK', 17), ('WAKEHAM JOHN', 17), ('WODRASKA JOHN', 17), ('GILLIS JOHN', 17), ('SCRIMSHAW MATTHEW', 17), ('CHAN RONNIE', 16), ('YEAP SOON', 16), ('CHRISTODOULOU DIOMEDES', 16), ('WINOKUR JR. HERBERT S', 16), ('MENDELSOHN JOHN', 16), ('BLAKE JR. NORMAN P', 16), ('URQUHART JOHN A', 16), ('PEREIRA PAULO V. FERRAZ', 16), ('LOWRY CHARLES P', 16), ('MEYER JEROME J', 16), ('GATHMANN WILLIAM D', 16), ('FUGH JOHN L', 16)]
# 
# According to the fact I found out earlier, the total of key-value pairs in each datum is 21, and POI identity must be identified with boolean (True or False). As the person "LOCKHART EUGENE E" have 20 NaN values, I would say this datum was unable to help us predict or figure out who might be the POI because there were no effective features. Additionally, although the persons with NaN value count higher than 13 were all non-POI, we might not be able to assure and predict that every person with high NaN value count were all non-POI, and lacking of information might make the prediction easy to be biased.
# 
# Before transforming the none values into numbers to easily analyze and run machine learning algorithms, I also check whether there were any zeros in the original dataset and I noticed that some data points had zeros besides the boolean value (False, non-POI). I would like to take this into consideration and turn the NaN values into -10. as a penalty value when doing the data transformation. Note that most value of feature "deferred_income" were negative, so I set the penalty to be positive.

# ### Step 2: Data pre-process and analysis

# In[57]:

def ExtractFeatureNames(dict):
    "Define a function to extract all features in the dict of dict and easier to plot"
    dict_values = list(dict.values())[0]
    dict_values_features = list(dict_values.keys())
    dict_values_features.remove('poi')
    dict_values_features.remove('email_address')
    dict_values_features = sorted(dict_values_features)
    #print(dict_values_features)
    return dict_values_features


# In[58]:

### Store to my_dataset for easy export below.
my_dataset = data_dict

#preprocess on the NaN value in the dictionary!!!
def RemoveNaN(dict, penalty, RemoveOutlier = False):
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
    if RemoveOutlier:
        dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
        dict.pop('TOTAL', 0)
        dict.pop('LOCKHART EUGENE E', 0)
    return dict
my_dataset = RemoveNaN(my_dataset, -10, RemoveOutlier = True)

def ResetFeatureList(my_dataset):
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi']
    features_list.extend(ExtractFeatureNames(my_dataset))
    return features_list
    #Total 20 features including poi as the first feature

features_list = ResetFeatureList(my_dataset)
#print(list(my_dataset.items())[1])
print(features_list)


# After removing the "NaN" values, I created a features list to store all features in the dataset. This would help a lot when I needed to put features into any algorithms to analyze.

# ###Outliers
# In addition, by checking the pdf file of financial benefits, I noticed that there were 2 clear outliers in the data, "TOTAL" and "THE TRAVEL AGENCY IN THE PARK". The first one seemed to be the sum total of all the other data points, while the second outlier was quite bizarre. Both these outliers and the datum with all NaN values, "LOCKHART EUGENE E", were removed from the dataset for all the analysis by applying __dict.pop(key, 0)__. 

# In[59]:

def is_nan(x):
    return (x is np.nan or x != x)
#create a dict that key is the feature name and values are a list of floats

def DictWithFeature(dataset, sorting=False):
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


# In[60]:

dict_with_features = DictWithFeature(my_dataset)
from scipy.stats.stats import pearsonr
for key in dict_with_features.keys():
    print ('PearsonR between poi and ', key, " : ", 
           pearsonr(dict_with_features['poi'], dict_with_features[key]))


# ###Correlation between POI label and others
# To better predict who is more likely to be a POI, first of all I computed the correlation between label "poi" and the others, which was listed below:
# 
#  | Label 1 |  Label 2 |  Pearson's r |      
#  | --- | --- | --- |
#  | poi | exercised_stock_options | 0.387 |
#  | poi | total_payments | 0.242 |
#  | poi | expenses | 0.204 |
#  | poi | deferral_payments | -0.040 |
#  | poi | to_messages | 0.108 |
#  | poi | other | 0.170 |
#  | poi | restricted_stock | 0.248 |
#  | poi | bonus | 0.358 |
#  | poi | total_stock_value | 0.383 |
#  | poi | restricted_stock_deferred | -0.022 |
#  | poi | loan_advances | 0.220 |
#  | poi | shared_receipt_with_poi | 0.240 |
#  | poi | from_this_person_to_poi | 0.135 |
#  | poi | long_term_incentive | 0.256 |
#  | poi | from_messages | -0.034 |
#  | poi | from_poi_to_this_person | 0.193 |
#  | poi | deferred_income | -0.274 |
#  | poi | salary | 0.339 |
#  | poi | director_fees | -0.122 |
# 
# According to the table, I noticed that there were no any strong relationships between the label "poi" and the others. Since I could not simply classify which labels had more impacts on making a person become a POI, I would like to add more computational labels and apply PCA, which stands for principal components analysis, to pick up the most related labels in the dataset.

# ### Step 3: Adding new features

# ##New Features
# From the initial dataset, Some new features were added, you can find more details in the table below:
# 
# 
#  Feature        |  Description             
#  :--- | --- 
#  Ratio of messages received from POI       | messages received from POI divided by total received messages
#  Ratio of messages sent to POI    | messages sent to POI divided by total sent messages
#  Comparison to Average (squared)      | features (financial and non-financial) divided by the average amount of the dataset          
#  Comparison to Median (squared) | features (financial and non-financial) divided by the median amount (if not zero) of the dataset
# 
# The reason behind the new features of message ratio created was that I expected that POI contacted with each other relatively more often than non-POI and the relationship might be non-linear. To enlarge the variance, I would like to squaring all the new features as well. I also expected that the financial gains of POI are more than the average and median, that was why I compared each feature with the average and median and squared it to get bigger variance.

# In[61]:

#Create new features about ratio of each feature to the mean and median of the feature
mean_dict, median_dict = dict(), dict()
#Create a dict to know each mean of feature
for key, value in dict_with_features.items():
    mean_dict[key] = sum(value)/len(value)
    median_dict[key] = sorted(value)[int(len(value)/2)]
print(mean_dict, median_dict)


# In[62]:

#Create feature list
for point in my_dataset.keys():
    for key in features_list:
        try: mean_dict[key]
        except: continue
        if key != 'poi' and mean_dict[key] != 0:
            my_dataset[point]['%s_mean_ratio' % key] =             my_dataset[point][key] / mean_dict[key]
        if key != 'poi' and median_dict[key] != 0:
            my_dataset[point]['%s_median_ratio' % key] =             my_dataset[point][key] / median_dict[key]
#check
#print(my_dataset['KISHKILL JOSEPH G'])


# In[63]:

#Create new features about square of each feature
#import re
#for point in my_dataset.keys():
#    for key in features_list:
#        if re.findall("_sqr", key) or key == 'poi': continue
#        else:
#            my_dataset[point]['%s_sqr' % key] = \
#                my_dataset[point][key] **2


# In[64]:

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
def PlottingFunction(dictionary, value1, value2, threshold1, threshold2, threshold_option = False):
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
PlottingFunction(my_dataset, 'from_poi_message_ratio', 'to_poi_message_ratio', 0.0004 , 0.03, True )



# ![](poi_messages.png)
# After computing the ratio of messages received from poi to total received messages and messages sending to poi to total sent messages, I saw that the filter excluded 72 persons before excluded any POI wrongly. It seems like these two features will help me somehow when creating the machine learning classifier. Since I couldn't find any new features to describe data distribution better, next I would like to try some of machine learning algorithms and see which one performed better.

# ### Step 4: Choosing machine learning algorithm

# ##Algorithms selection and tuning
# For the analysis of the data, a total of 6 classifiers was applied, which included:
# 
# ####Decision Tree Classifier
# ####Gaussian Naive Bayes
# ####Support Vector Classifier (SVC)
# ####AdaBoost
# ####Random Forrest Tree Classifier
# ####K Nearest Neighbor
# 
# The object of the algorithm was to classify and find out which people are more likely to be a POI. There were clearly 2 categories I was looking to label the data.
# 
# To tune the algorithm, I applied PCA to decompose features and dimentions, and MaxAbsScaler to scale and normalize the features. Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components (or sometimes, principal modes of variation). The number of principal components is less than or equal to the smaller of the number of original variables or the number of observations. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. I did not think the outcome would be the best when I put most features into the algorithms, either only few ones. Also for avoiding overfitting and underfitting, I would like to try PCA components between 2 and 20 since there were 58 features in total in the dataset. MaxAbsScaler transforms a dataset of Vector rows, rescaling each feature to range [-1, 1] by dividing through the maximum absolute value in each feature. It does not shift/center the data, and thus does not destroy any sparsity. MaxAbsScaler computes summary statistics on a data set and produces a MaxAbsScalerModel. The model can then transform each feature individually to range [-1, 1].
# 
# For the most part, PCA made a huge improvement when trying different feature numbers. You might find more comparison details from the executing section.

# In[65]:

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


# In[66]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
    
data = featureFormat(my_dataset, ResetFeatureList(my_dataset), sort_keys = True)
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


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
#from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit


# In[67]:

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
                features_train.append( features[ii] )
                labels_train.append( labels[ii] )
            for jj in test_idx:
                features_test.append( features[jj] )
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


# In[68]:

algorithms_comparison = dict()
algorithms_comparison['Naive Bayes'] = algorithm_tester(
    GaussianNB(), 100, [2, 21])


# In[ ]:

algorithms_comparison['SVM'] = algorithm_tester(
    SVC(), 100, [2, 21])


# In[ ]:

algorithms_comparison['Decision Tree'] = algorithm_tester(
    DecisionTreeClassifier(), 100, [2, 21] )


# In[ ]:

algorithms_comparison['Random Forest'] = algorithm_tester(
    RandomForestClassifier(), 100,[2, 21])


# In[ ]:

algorithms_comparison['AdaBoost'] = algorithm_tester(
    AdaBoostClassifier(), 100, [2, 21])


# In[ ]:

algorithms_comparison['K Nearest Neighbors'] = algorithm_tester(
    KNeighborsClassifier(), 100, [2, 21])


# In[ ]:

print(algorithms_comparison)


# By running the machine learning codes, I noticed that every algorithm return high accuracy, but does that mean the prediction is good? Or that just results from the low ratio of persons of interest to all people in the dataset?
# After importing and computing recall and precision scores, I found that some algorithms get really bad recall and precision scores. In addition, after testing multiple times, I found that only Naive Bayes returned  recall and precision scores both higher than 0.3. Additionally, there were many different n_component choices in PCA process when running Naive Bayes as the chosen algorithm. Here are the algorithms and PCA choices with both recall and precision scores higher than 0.3:
# 
#  | ML Method |  PCA n_components |  Recall Score |  Precision Score |      
#  | --- | --- | --- | --- |
#  | Naive Bayes GaussianNB | 2 | 0.330 | 0.584 | 
#  | Naive Bayes GaussianNB | 3 | 0.330 | 0.555 | 
#  | Naive Bayes GaussianNB | 4 | 0.305 | 0.513 | 
#  | Naive Bayes GaussianNB | 7 | 0.330 | 0.443 | 
#  | Naive Bayes GaussianNB | 8 | 0.390 | 0.446 | 
#  | Naive Bayes GaussianNB | 9 | 0.330 | 0.410 | 
#  | Naive Bayes GaussianNB | 10 | 0.305 | 0.415 |
#  | Naive Bayes GaussianNB | 11 | 0.315 | 0.444 |
#  | Naive Bayes GaussianNB | 12 | 0.310 | 0.437 |
#  | Decision Tree | 3 | 0.335 | 0.310 |
#  | Decision Tree | 5 | 0.310 | 0.302 |
#  |Results WILL vary. There is some randomness in the data splitting |  
# 
# From this table we might find that applying algorithm Naive Bayes GaussianNB with 2 to 4 and 7 to 9 PCA n_components would returned the best outcome.

# In[44]:

best_outcome = algorithm_tester(GaussianNB(), 1000, [2, 13])
print(best_outcome)


# ##Validation and Performance
# 
# To validate the performance of each algorithm, **recall** and **precision** scores were calculated for each one. The scores of the best algorithm were listed below:
# 
#  |Algorithm        |  PCA n_component |  Recall | Precision |           
#  |:--- | --- | --- | --- | 
#  |Naive Bayes GaussianNB | 8 | 0.347 | 0.426 | 
#  |Results WILL vary. There is some randomness in the data splitting 
#  
# The best classifier was actually *Naive Bayes GaussianNB* using PCA beforehand. This was achieved by using sklearn Pipline. The GaussianNB achieved a consistent score above 0.30 for both precision and recall. The final parameters applied are detailed below:
# 
# ##### Pipeline([('reduce_dim', PCA(n_components = 9)), ('clf', GaussianNB())])
# 
# 

# ##Discussion and Conclusions
# 
# ###New features impact
# 
# When conducting this project, I added a lot of new features. I'd like to compare what each algorithm returned with no any new features added or even no penalty for the NaN values. 

# In[45]:

#try if we don't add any new features
with open("/Users/KunWuYao/GitHub/Udacity_ML_projects/final_project/final_project_dataset.pkl", "rb") as data_file:
    my_dataset_o = pickle.load(data_file)
my_dataset_o = RemoveNaN(my_dataset_o, 0., RemoveOutlier = True)
features_list = ResetFeatureList(my_dataset_o)
data_o = featureFormat(my_dataset_o, features_list, sort_keys = True)
labels, features = targetFeatureSplit( data_o ) 
features_normalized = min_max_scaler.fit_transform(features)


# In[46]:

algorithms_comparison_o = dict()
algorithms_comparison_o['Naive Bayes'] = algorithm_tester(GaussianNB(),100, [2, 20])


# In[47]:

algorithms_comparison_o['SVM'] = algorithm_tester(SVC(),100,[2, 20])


# In[48]:

algorithms_comparison_o['Decision Tree'] = algorithm_tester(DecisionTreeClassifier(),100,[2, 20])


# In[49]:

algorithms_comparison_o['Random Forest'] = algorithm_tester(RandomForestClassifier(),100, [2, 20])


# In[50]:

algorithms_comparison_o['AdaBoost'] = algorithm_tester(AdaBoostClassifier(),100, [2, 20])


# In[51]:

algorithms_comparison_o['K Nearest Neighbors'] = algorithm_tester(KNeighborsClassifier(),100, [2, 20])


# In[52]:

print(algorithms_comparison_o)


# In[53]:

best_outcome_o = (algorithm_tester(GaussianNB(),1000, [2, 10]))
print(best_outcome_o)


# After running to get the consequences, I found that after lots of new features were added, the best score was only a little higher, and that made me think maybe PCA was not the best method to find out how many and which features I shall pick up. This was just a starting point analysis for classifying Enron employees. The results should not be taken too seriously and more advanced models should be used. Possibilities for future research could be to include more complex pipelines for the data, try different feature selecting tools and learning parameters, or even Neural Networks.

# ##References
# 
# [Udacity - Intro to Machine Learning course] [0]
# 
# [Sklearn documentation] [1]
# 
# [0]: https://www.udacity.com/course/intro-to-machine-learning--ud120
# [1]: http://scikit-learn.org/stable/documentation.html 

# In[70]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
from tester import dump_classifier_and_data
clf = Pipeline([('reduce_dim', PCA(n_components = 8)), ('clf', GaussianNB())])
features_list = ResetFeatureList(my_dataset)


# In[17]:

# Finally, output the data
dump_classifier_and_data(clf, my_dataset, features_list)

