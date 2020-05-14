import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#----------------------------------------------------------------------------------------
# ####################### ITERATION - 1 ################################################

# USE FULL DATASET FOR TRAINING ALGORITHM
#----------------------------------------------------------------------------------------
# ############################# Importing the dataset ################################################
winequalityfile = r'D:\DEEPA\Data Science\Spyder codes\winequality-red.csv'
dataset = pd.read_csv(winequalityfile)
print(dataset)


# ############################# Exploring data analysis (EDA) using functions #######################################
dataset.shape # 1599,12
dataset.columns
dataset.isnull().sum() # To check null values
dataset.info() #last column - quality is the target column (int)

quality = dataset['quality'].unique()
print(sorted(quality))

dataset.describe().T
dataset.groupby('quality').describe().T

#Checking class imbalance
dataset.groupby('quality').size()


# ############################# Exploring data Analysis (EDA) using graphs (Data Visualization) #####################

# 1) Box plot for all columns - group by "quality" column
for col_name in dataset.columns:
    print(col_name)
    statement = "dataset.boxplot(column = '" + col_name + "', by = 'quality')"
    exec(statement)    
    plt.show()

# 2) Histogram for all columns - group by "quality" column 
# Note : Only to know distribution, use this
#        Not much useful as target column has many categories
for col_name in dataset.columns:
    print(col_name)
    statement = "dataset.hist(column = '" + col_name +"', by = 'quality');"
    exec(statement)    
    plt.show()

# 3) Scatter Matrix - Best to know relation
scatter_matrix(dataset,figsize=(8,8))
plt.show()


# ############################# To know co-relation ##################################################
corr_matrix = dataset.corr()
print(corr_matrix)
print(corr_matrix["quality"].sort_values(ascending=False))

# ############################# Splitting the data into train and test dataset ######################
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ############################# Applying Machine Learning Algorithm ################################

# ############################# Applying LINEAR REGRESSION

print('---------- Linear Regression --------------------------')

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df

# NOTE : COEFFICIENT IS THE MAIN FACTOR OF TARGET COLUMN, IT SHOWS HOW MUCH IT WILL CHANGE
# NOTE : CORELATION SHOWS THE RELATION BETWEEN TARGET AND FEATURE COLUMN

# BOTH (COEFICIENT AND COREALATION) NEED TO BE CONSIDER TO KNOW WHICH COLUMNS TO DROP IN ALGORITHM

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()

#Performance evaluation
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R-squared score:', regressor.score(X_test, y_test))  

# Prepare classification report and confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, np.round(y_pred,0)))
print(confusion_matrix(y_test, np.round(y_pred,0)))
print(classification_report(y_test, np.round(y_pred,0)))

iteration1 = accuracy_score(df.Actual,df.Predicted)

#Which is the best approach? Classification or regression?

# ############################# Applying CLASSIFICATION

#------------------- Machine learning 2 ------------
print('---------- KNeighborsClassifier--------------------------')
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 3 ------------
print('---------- DecisionTreeClassifier--------------------------')

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 4 ------------
print('---------- SVC--------------------------')

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 5 ------------
print('---------- GaussianNB--------------------------')

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 6 ------------
print('---------- RandomForestClassifier--------------------------')

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 7 ------------
print('---------- MLPClassifier--------------------------')

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------
# Result - RandomForestClassifier is best for algorithm
#----------------------------------------------------------------------------------------


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#----------------------------------------------------------------------------------------
# ####################### ITERATION - 2 ################################################

# REMOVE UNWANTED COLUMNS AND APPLY MACHINE LEARNING ALGORITHM AGAIN
#----------------------------------------------------------------------------------------

dataset = pd.read_csv(winequalityfile)
print(dataset)
dataset.columns

# To analysis unwanted columns ###########################################################

#sns.barplot(x = 'quality', y = 'fixed acidity', data = dataset) #does not look helpful
for col_name in dataset.columns:
    print(col_name)
    statement = "sns.barplot(x = 'quality', y = '" + col_name +"', data = dataset)"    
    exec(statement)    
    plt.show()
    
#fixed acidity - does not look helpful

#volatile acidity - quality increases as volatile acidity decreases

#citric acid - quality increases as citric acid increases

#residual sugar - #does not look helpful

#chlorides - quality increases as chlorides decreases

#free sulfur dioxide - #indecisive

#total sulfur dioxide - #indecisive

#density - #does not look helpful

#pH - #does not look helpful

#sulphates - #quality increases as sulphates increases

#alcohol - quality increases as alcohol increases
    
# To remove unwanted columns #############################################################

dataset1 = dataset.drop(['fixed acidity','residual sugar','density','pH'], axis=1)
dataset1.columns

# ############################# Splitting the data into train and test dataset ######################
X = dataset1.iloc[:,:-1].values
y = dataset1.iloc[:,-1].values

#standardizing the data
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ############################# Applying Machine Learning Algorithm ################################

# ############################# Applying LINEAR REGRESSION

print('---------- Linear Regression --------------------------')

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['volatile acidity', 'citric acid',  
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df

# NOTE : COEFFICIENT IS THE MAIN FACTOR OF TARGET COLUMN, IT SHOWS HOW MUCH IT WILL CHANGE
# NOTE : CORELATION SHOWS THE RELATION BETWEEN TARGET AND FEATURE COLUMN

# BOTH (COEFICIENT AND COREALATION) NEED TO BE CONSIDER TO KNOW WHICH COLUMNS TO DROP IN ALGORITHM

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()

#Performance evaluation
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R-squared score:', regressor.score(X_test, y_test))  

# Prepare classification report and confusion matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, np.round(y_pred,0)))
print(confusion_matrix(y_test, np.round(y_pred,0)))
print(classification_report(y_test, np.round(y_pred,0)))

iteration2 = accuracy_score(df.Actual,df.Predicted)

#Which is the best approach? Classification or regression?

# ############################# Applying CLASSIFICATION

#------------------- Machine learning 2 ------------
print('---------- KNeighborsClassifier--------------------------')
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 3 ------------
print('---------- DecisionTreeClassifier--------------------------')

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 4 ------------
print('---------- SVC--------------------------')

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 5 ------------
print('---------- GaussianNB--------------------------')

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 6 ------------
print('---------- RandomForestClassifier--------------------------')

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#------------------- Machine learning 7 ------------
print('---------- MLPClassifier--------------------------')

from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier()
classifier.fit(X_train, y_train)

accuracy = round(classifier.score(X_test, y_test) * 100, 2)
print(accuracy)

#predict the class of wine for y_train
y_pred = classifier.predict(X_test)
# Evaluate predictions
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------
# Result - RandomForestClassifier is best for algorithm - accuracy increased
#----------------------------------------------------------------------------------------

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#----------------------------------------------------------------------------------------
# ####################### ITERATION - 3 ################################################

# REMOVE UNWANTED COLUMNS AND INDECISIVE COLUMNS AND APPLY MACHINE LEARNING ALGORITHM AGAIN
#----------------------------------------------------------------------------------------
dataset = pd.read_csv(winequalityfile)
print(dataset)
dataset.columns

# To analysis unwanted columns ###########################################################

#sns.barplot(x = 'quality', y = 'fixed acidity', data = dataset) #does not look helpful
for col_name in dataset.columns:
    print(col_name)
    statement = "sns.barplot(x = 'quality', y = '" + col_name +"', data = dataset)"    
    exec(statement)    
    plt.show()
    
#fixed acidity - does not look helpful

#volatile acidity - quality increases as volatile acidity decreases

#citric acid - quality increases as citric acid increases

#residual sugar - #does not look helpful

#chlorides - quality increases as chlorides decreases

#free sulfur dioxide - #indecisive

#total sulfur dioxide - #indecisive

#density - #does not look helpful

#pH - #does not look helpful

#sulphates - #quality increases as sulphates increases

#alcohol - quality increases as alcohol increases
    
# To remove unwanted columns #############################################################

dataset1 = dataset.drop(['fixed acidity','residual sugar','density','pH'], axis=1)
dataset1.columns    

#let's drop the indecisive ones too

dataset2 = dataset1.drop(['free sulfur dioxide','total sulfur dioxide'], axis=1)
dataset2.columns   

X = dataset2.iloc[:,:-1].values
y = dataset2.iloc[:,-1].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#Exploring the effets of all the variables
variables = ['volatile acidity', 'citric acid', 'chlorides', 'sulphates','alcohol']

coeff_df = pd.DataFrame(regressor.coef_, index=variables, columns=['Coefficient'])  
coeff_df #compare it with the above graphs

#Predicting the test set values
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred,0)}, dtype=int)
df.head()
df.tail()

#Performance evaluation
print('R-squared score:', regressor.score(X_test, y_test))  

confusion_matrix(df.Actual,df.Predicted)
iteration3 = accuracy_score(df.Actual,df.Predicted) 

###############################################################################################
#                FINAL RESULT
###############################################################################################

# PRINTING ACCURACY OF ALL ITERATIONS
print(round(iteration1,2), round(iteration2,2), round(iteration3,2))