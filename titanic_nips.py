# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:21:17 2021

@author: nicol
"""

# %% [markdown]
# # Titanic Data Science Solutions
# 
# 
# ### This notebook is a companion to the book [Data Science Solutions](https://www.amazon.com/Data-Science-Solutions-Startup-Workflow/dp/1520545312). 
# 
# The notebook walks us through a typical workflow for solving data science competitions at sites like Kaggle.
# 
# There are several excellent notebooks to study data science competition entries. However many will skip some of the explanation on how the solution is developed as these notebooks are developed by experts for experts. The objective of this notebook is to follow a step-by-step workflow, explaining each step and rationale for every decision we take during solution development.
# 
# ## Workflow stages
# 
# The competition solution workflow goes through seven stages described in the Data Science Solutions book.
# 
# 1. Question or problem definition.
# 2. Acquire training and testing data.
# 3. Wrangle, prepare, cleanse the data.
# 4. Analyze, identify patterns, and explore the data.
# 5. Model, predict and solve the problem.
# 6. Visualize, report, and present the problem solving steps and final solution.
# 7. Supply or submit the results.
# 
# The workflow indicates general sequence of how each stage may follow the other. However there are use cases with exceptions.
# 
# - We may combine mulitple workflow stages. We may analyze by visualizing data.
# - Perform a stage earlier than indicated. We may analyze data before and after wrangling.
# - Perform a stage multiple times in our workflow. Visualize stage may be used multiple times.
# - Drop a stage altogether. We may not need supply stage to productize or service enable our dataset for a competition.
# 
# 
# ## Question and problem definition
# 
# Competition sites like Kaggle define the problem to solve or questions to ask while providing the datasets for training your data science model and testing the model results against a test dataset. The question or problem definition for Titanic Survival competition is [described here at Kaggle](https://www.kaggle.com/c/titanic).
# 
# > Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
# 
# We may also want to develop some early understanding about the domain of our problem. This is described on the [Kaggle competition description page here](https://www.kaggle.com/c/titanic). Here are the highlights to note.
# 
# - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# - One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# ## Workflow goals
# 
# The data science solutions workflow solves for seven major goals.
# 
# **Classifying.** We may want to classify or categorize our samples. We may also want to understand the implications or correlation of different classes with our solution goal.
# 
# **Correlating.** One can approach the problem based on available features within the training dataset. Which features within the dataset contribute significantly to our solution goal? Statistically speaking is there a [correlation](https://en.wikiversity.org/wiki/Correlation) among a feature and solution goal? As the feature values change does the solution state change as well, and visa-versa? This can be tested both for numerical and categorical features in the given dataset. We may also want to determine correlation among features other than survival for subsequent goals and workflow stages. Correlating certain features may help in creating, completing, or correcting features.
# 
# **Converting.** For modeling stage, one needs to prepare the data. Depending on the choice of model algorithm one may require all features to be converted to numerical equivalent values. So for instance converting text categorical values to numeric values.
# 
# **Completing.** Data preparation may also require us to estimate any missing values within a feature. Model algorithms may work best when there are no missing values.
# 
# **Correcting.** We may also analyze the given training dataset for errors or possibly innacurate values within features and try to corrent these values or exclude the samples containing the errors. One way to do this is to detect any outliers among our samples or features. We may also completely discard a feature if it is not contribting to the analysis or may significantly skew the results.
# 
# **Creating.** Can we create new features based on an existing feature or a set of features, such that the new feature follows the correlation, conversion, completeness goals.
# 
# **Charting.** How to select the right visualization plots and charts depending on nature of the data and the solution goals.

# %% [markdown]
# ## Refactor Release 2017-Jan-29
# 
# We are significantly refactoring the notebook based on (a) comments received by readers, (b) issues in porting notebook from Jupyter kernel (2.7) to Kaggle kernel (3.5), and (c) review of few more best practice kernels.
# 
# ### User comments
# 
# - Combine training and test data for certain operations like converting titles across dataset to numerical values. (thanks @Sharan Naribole)
# - Correct observation - nearly 30% of the passengers had siblings and/or spouses aboard. (thanks @Reinhard)
# - Correctly interpreting logistic regresssion coefficients. (thanks @Reinhard)
# 
# ### Porting issues
# 
# - Specify plot dimensions, bring legend into plot.
# 
# 
# ### Best practices
# 
# - Performing feature correlation analysis early in the project.
# - Using multiple plots instead of overlays for readability.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# %% [markdown]
# ## Acquire data
# 
# The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df = pd.read_csv('../Data/titanic/train.csv')
test_df = pd.read_csv('../Data/titanic/test.csv')
combine = [train_df, test_df]

# %% [markdown]
# ## Analyze by describing data
# 
# Pandas also helps describe the datasets answering following questions early in our project.
# 
# **Which features are available in the dataset?**
# 
# Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data).

# %% [code] {"jupyter":{"outputs_hidden":true}}
print(train_df.columns.values)

# %% [markdown]
# **Which features are categorical?**
# 
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# 
# **Which features are numerical?**
# 
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Continous: Age, Fare. Discrete: SibSp, Parch.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# preview the data
train_df.head()

# %% [markdown]
# **Which features are mixed data types?**
# 
# Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
# 
# - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# **Which features may contain errors or typos?**
# 
# This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
# 
# - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df.tail()

# %% [markdown]
# **Which features contain blank, null or empty values?**
# 
# These will require correcting.
# 
# - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# - Cabin > Age are incomplete in case of test dataset.
# 
# **What are the data types for various features?**
# 
# Helping us during converting goal.
# 
# - Seven features are integer or floats. Six in case of test dataset.
# - Five features are strings (object).

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df.info()
print('_'*40)
test_df.info()

# %% [markdown]
# **What is the distribution of numerical feature values across the samples?**
# 
# This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
# 
# - Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
# - Survived is a categorical feature with 0 or 1 values.
# - Around 38% samples survived representative of the actual survival rate at 32%.
# - Most passengers (> 75%) did not travel with parents or children.
# - Nearly 30% of the passengers had siblings and/or spouse aboard.
# - Fares varied significantly with few passengers (<1%) paying as high as $512.
# - Few elderly passengers (<1%) within age range 65-80.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

# %% [markdown]
# **What is the distribution of categorical features?**
# 
# - Names are unique across the dataset (count=unique=891)
# - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# - Embarked takes three possible values. S port used by most passengers (top=S)
# - Ticket feature has high ratio (22%) of duplicate values (unique=681).

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df.describe(include=['O'])

# %% [markdown]
# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# **Correlating.**
# 
# We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# **Completing.**
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# **Correcting.**
# 
# 1. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
# 2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 3. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# **Creating.**
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# **Classifying.**
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived. 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.

# %% [markdown]
# ## Analyze by pivoting features
# 
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# 
# - **Pclass** We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# - **Sex** We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# - **SibSp and Parch** These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [markdown]
# ## Analyze by visualizing data
# 
# Now we can continue confirming some of our assumptions using visualizations for analyzing the data.
# 
# ### Correlating numerical features
# 
# Let us start by understanding correlations between numerical features and our solution goal (Survived).
# 
# A histogram chart is useful for analyzing continous numerical variables like Age where banding or ranges will help identify useful patterns. The histogram can indicate distribution of samples using automatically defined bins or equally ranged bands. This helps us answer questions relating to specific bands (Did infants have better survival rate?)
# 
# Note that x-axis in historgram visualizations represents the count of samples or passengers.
# 
# **Observations.**
# 
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# 
# **Decisions.**
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# 
# - We should consider Age (our assumption classifying #2) in our model training.
# - Complete the Age feature for null values (completing #1).
# - We should band age groups (creating #3).

# %% [code] {"jupyter":{"outputs_hidden":true}}
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# %% [markdown]
# ### Correlating numerical and ordinal features
# 
# We can combine multiple features for identifying correlations using a single plot. This can be done with numerical and categorical features which have numeric values.
# 
# **Observations.**
# 
# - Pclass=3 had most passengers, however most did not survive. Confirms our classifying assumption #2.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies our classifying assumption #2.
# - Most passengers in Pclass=1 survived. Confirms our classifying assumption #3.
# - Pclass varies in terms of Age distribution of passengers.
# 
# **Decisions.**
# 
# - Consider Pclass for model training.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# %% [markdown]
# ### Correlating categorical features
# 
# Now we can correlate categorical features with our solution goal.
# 
# **Observations.**
# 
# - Female passengers had much better survival rate than males. Confirms classifying (#1).
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
# 
# **Decisions.**
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# %% [markdown]
# ### Correlating categorical and numerical features
# 
# We may also want to correlate categorical features (with non-numeric values) and numeric features. We can consider correlating Embarked (Categorical non-numeric), Sex (Categorical non-numeric), Fare (Numeric continuous), with Survived (Categorical numeric).
# 
# **Observations.**
# 
# - Higher fare paying passengers had better survival. Confirms our assumption for creating (#4) fare ranges.
# - Port of embarkation correlates with survival rates. Confirms correlating (#1) and completing (#2).
# 
# **Decisions.**
# 
# - Consider banding Fare feature.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# %% [markdown]
# ## Wrangle data
# 
# We have collected several assumptions and decisions regarding our datasets and solution requirements. So far we did not have to change a single feature or value to arrive at these. Let us now execute our decisions and assumptions for correcting, creating, and completing goals.
# 
# ### Correcting by dropping features
# 
# This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
# 
# Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
# 
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

# %% [code] {"jupyter":{"outputs_hidden":true}}
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

# %% [markdown]
# ### Creating new feature extracting from existing
# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
# 
# **Observations.**
# 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# **Decision.**
# 
# - We decide to retain the new Title feature for model training.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

# %% [markdown]
# We can replace many titles with a more common name or classify them as `Rare`.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# %% [markdown]
# We can convert the categorical titles to ordinal.

# %% [code] {"jupyter":{"outputs_hidden":true}}
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# %% [markdown]
# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# %% [markdown]
# ### Converting a categorical feature
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# %% [markdown]
# ### Completing a numerical continuous feature
# 
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# 1. A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
# 
# 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
# 3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# %% [markdown]
# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

# %% [code] {"jupyter":{"outputs_hidden":true}}
guess_ages = np.zeros((2,3))
guess_ages

# %% [markdown]
# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

# %% [markdown]
# Let us create Age bands and determine correlations with Survived.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# %% [markdown]
# Let us replace Age with ordinals based on these bands.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df.head()

# %% [markdown]
# We can not remove the AgeBand feature.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

# %% [markdown]
# ### Create new feature combining existing features
# 
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [markdown]
# We can create another feature called IsAlone.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# %% [markdown]
# Let us drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

# %% [markdown]
# We can also create an artificial feature combining Pclass and Age.

# %% [code] {"jupyter":{"outputs_hidden":true}}
"""
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
"""
# %% [markdown]
# ### Completing a categorical feature
# 
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

# %% [code] {"jupyter":{"outputs_hidden":true}}
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# %% [markdown]
# ### Converting categorical feature to numeric
# 
# We can now convert the EmbarkedFill feature by creating a new numeric Port feature.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

# %% [markdown]
# ### Quick completing and converting a numeric feature
# 
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
# 
# Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
# 
# We may also want round off the fare to two decimals as it represents currency.

# %% [code] {"jupyter":{"outputs_hidden":true}}
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

# %% [markdown]
# We can not create FareBand.

# %% [code] {"jupyter":{"outputs_hidden":true}}
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# %% [markdown]
# Convert the Fare feature to ordinal values based on the FareBand.

# %% [code] {"jupyter":{"outputs_hidden":true}}
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)

# %% [markdown]
# And the test dataset.

# %% [code] {"jupyter":{"outputs_hidden":true}}
test_df.head(10)

# %% [markdown]
# ## Model, predict and solve
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forrest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine

# %% [code] {"jupyter":{"outputs_hidden":true}}
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# %% [markdown]
# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# %% [markdown]
# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - Inversely as Pclass increases, probability of Survived=1 decreases the most.
# - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# - So is Title as second highest positive correlation.

# %% [code] {"jupyter":{"outputs_hidden":true}}
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# %% [markdown]
# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of **two categories**, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
# 
# Note that the model generates a confidence score which is higher than Logistics Regression model.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

# %% [markdown]
# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
# 
# KNN confidence score is better than Logistics Regression but worse than SVM.

# %% [code] {"jupyter":{"outputs_hidden":true}}
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# %% [markdown]
# In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. Reference [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).
# 
# The model generated confidence score is the lowest among the models evaluated so far.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

# %% [markdown]
# The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time. Reference [Wikipedia](https://en.wikipedia.org/wiki/Perceptron).

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

# %% [markdown]
# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
# 
# The model confidence score is the highest among models evaluated so far.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# %% [markdown]
# The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
# 
# The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

# %% [code] {"jupyter":{"outputs_hidden":true}}
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# %% [markdown]
# ### Model evaluation
# 
# We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set. 

# %% [code] {"jupyter":{"outputs_hidden":true}}
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

# %% [code] {"jupyter":{"outputs_hidden":true}}
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

# %% [markdown]
# Our submission to the competition site Kaggle results in scoring 3,883 of 6,082 competition entries. This result is indicative while the competition is running. This result only accounts for part of the submission dataset. Not bad for our first attempt. Any suggestions to improve our score are most welcome.

# %% [markdown]
# ## References
# 
# This notebook has been created based on great work done solving the Titanic competition and other sources.
# 
# - [A journey through Titanic](https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic)
# - [Getting Started with Pandas: Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)
# - [Titanic Best Working Classifier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)



df = pd.DataFrame(random_forest.feature_importances_, index=X_train.columns)
df.plot(kind='bar')


from sklearn import tree
tree.plot_tree(decision_tree)







feature_names=X_train.columns

X_train = X_train.values
y_train = Y_train


from sklearn.utils import shuffle
X_train, _, y_train = shuffle(X_train, X_train, y_train, random_state=123)



#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import copy
import matplotlib as mpl
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc
# from utilities import metrics
import seaborn as sns
from kneed import KneeLocator
from sklearn.metrics import roc_auc_score, accuracy_score
sns.set(font_scale=1.1)
# from utilities import load_datasets, load_genes, load_genes_overall#, load_genes_only_sex
from sklearn.metrics import silhouette_score
from sklearn.metrics import classification_report , confusion_matrix
import math
from pathlib import Path
import os
import sys
import copy
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam, SGD, RMSprop, Adadelta
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from keras.activations import relu
from keras.activations import sigmoid
from numpy.random import seed
import random
import matplotlib as mpl
import matplotlib.pyplot as pl
import pandas as pd
import copy
import numpy as np
import tensorflow as tf
siz = 12
mpl.rcParams['xtick.labelsize'] = siz
mpl.rcParams['axes.labelsize'] = siz
mpl.rcParams['font.size'] = siz
mpl.rcParams['axes.labelsize'] = siz
pd.get_option('display.max_columns')
random.seed(0)
 #seed(0)
tf.random.set_seed(0)



p = 2 # number of relu layer
n1 = 3
n2 = 2
n_neurons = n1 + n2

def nn_model(best_param_nn_l1=0, ratio_l1_l2=0, best_param_activity=0):
    mod = Sequential()
    # mod.add(Dropout(0.5))
    mod.add(Dense(n1, use_bias=True, input_dim=X_train.shape[1], 
                  activation='relu',######## #,# kernel_initializer=Constant(0.001),)
                       #kernel_regularizer=l1_l2(best_param_nn_l1)))#, ratio_l1_l2*best_param_nn_l1)))
                  kernel_regularizer=l2(best_param_nn_l1),
                  activity_regularizer=l1(best_param_activity)))#,# bias_regularizer=l1(best_param_nn),
                       #activity_regularizer=l2(best_param_nn),))
                       #kernel_initializer=Constant(0.0001) ))
#    mod.add(Dropout(0.5))
    mod.add(Dense(n2, use_bias=True, activation='relu'))#, kernel_initializer=Constant(0.001)))

    #mod.add(Dense(2, use_bias=True, activation='relu'))#, kernel_initializer=Constant(0.001)))
                       #kernel_regularizer=l1(best_param_nn),))# bias_regularizer=l1(best_param_nn),
                       #activity_regularizer=l1(best_param_nn),))
#    mod.add(Dropout(0.5))    
    
    mod.add(Dense(1, use_bias=True, activation='sigmoid'))
    
    
    #opt = SGD(lr=0.01) #
    #opt=SGD(lr=0.001)#learning_rate=0.01)
    opt=Adam()#delta()#(lr=0.001)#learning_rate=0.01)
    # opt=RMSprop()
    mod.compile(loss='binary_crossentropy', optimizer=opt, #rmsprop',#opt, #'sgd',###opt,
                metrics=['accuracy'])#, steps_per_execution=10)
#    model_nn.fit(X_train, y_train, epochs=10, class_weight=class_weights_dict)
#    model_nn.evaluate(X_train, y_train)
    return mod
    



"""
def lr_model(best_param_nn_l1=0, ratio_l1_l2=0, best_param_activity=0):
    mod = Sequential()
    mod.add(Dense(1, use_bias=True, activation='sigmoid'))
    opt=Adam()
    mod.compile(loss='binary_crossentropy', optimizer=opt, #rmsprop',#opt, #'sgd',###opt,
                metrics=['accuracy'])#, steps_per_execution=10)
    return mod

# keras_model = nn_model(best_param_nn_l1=0.1)#
keras_model = lr_model()#best_param_activity=0.001, best_param_nn_l1=0.1)

history = keras_model.fit(X_train, y_train, epochs=100, validation_split=0.1)
                          #batch_size=10)#, class_weight=class_weights_dict)
print('performance on train set: ', keras_model.evaluate(X_train, y_train))

df_lr = pd.DataFrame()    
df_lr['Feature'] = [nam for nam in feature_names]
df_lr['I'] = keras_model.weights[0].numpy()
df_lr['I_abs'] =  np.abs(df_lr['I'])
df_lr = df_lr.sort_values(['I_abs'], ascending=False)
df_lr.plot(x='Feature', y='I', kind='bar')
"""


"""
model = KerasClassifier(build_fn=nn_model)#, epochs=10, verbose=3)#, batch_size=50,
                        #validation_split=0.1)
params_l1 = np.logspace(-1, -3, 5).tolist()
params_ratio = np.linspace(0.1, 10, 5).tolist()
param_grid = dict(best_param_nn_l1=params_l1)#, ratio_l1_l2=params_ratio)# 0.0005, 0.0001], 
cv_i = StratifiedKFold(n_splits=10, random_state=0)# shuffle=True, random_state=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                    cv=cv_i, scoring='roc_auc')
grid_result = grid.fit(X_train, y_train, epochs=100)#, validation_split=0.1)
pl.figure('grid search')        
pl.plot(params_l1, grid_result.cv_results_['mean_test_score']); pl.xscale('log')

pl.errorbar(params_l1, grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], marker='o', linestyle='-'); pl.xscale('log')
""" 


#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

#params_l1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]#np.logspace(0, -2, 50).tolist()


"""
params_l1 = np.logspace(-1, -2, 10).tolist()[::-1]
tes = []
for pp in params_l1:
    print(pp)
    nn = nn_model(best_param_nn_l1=pp)
    nn.fit(X_train, y_train, epochs=500)#, callbacks=[callback])#, validation_data=(X_test, y_test))
    tes.append(nn.evaluate(X_test, y_test)[1])
pl.plot(params_l1, tes)
pl.xscale('log')

"""


#best_p = {'best_param_nn_l1':0.00001, 'best_param_activity':0}#grid_result.best_params_
"""
best_p = grid_result.best_params_
keras_model = nn_model(**best_p)
""" 




# keras_model = nn_model(best_param_nn_l1=0.1)#
keras_model = nn_model(best_param_nn_l1=0.1)#best_param_activity=0.001, 

history = keras_model.fit(X_train, y_train, epochs=100, validation_split=0.1)
                          #batch_size=10)#, class_weight=class_weights_dict)
print('performance on train set: ', keras_model.evaluate(X_train, y_train))

pl.figure('loss as a function of epochs')
pl.plot(history.history['loss'], label='training')
pl.plot(history.history['val_loss'], label='validation')
pl.legend()

pl.figure('accuracy as a function of epochs')
pl.plot(history.history['accuracy'], label='training')
pl.plot(history.history['val_accuracy'], label='validation')
pl.legend()

df_fi = pd.DataFrame(history.model.weights[0].numpy()[:, 0], index=feature_names)

# y_test = df_task.loc[test_index_m]['target_variable'].values


# print('performance on test set: ', keras_model.evaluate(X_test, Y_test))



# collect weights and bias
W = []
for i_relu_layer in np.arange(p+2)[::2]:
    W.append(keras_model.weights[i_relu_layer].numpy())
W.append(keras_model.weights[p+2].numpy())

B = []
for i_relu_layer in np.arange(1,p+3)[::2]:
    B.append(np.expand_dims(keras_model.weights[i_relu_layer].numpy(), 0))
B.append(keras_model.weights[(p*2)+1].numpy())

# compute activations/z
A = []
Z = []
for k in range(p):
    if k == 0:
        A.append(np.dot(X_train, W[0]) + B[0])
        Z.append(relu(A[0]).numpy())
    else:
        A.append(np.dot(Z[k-1], W[k]) + B[k])
        Z.append(relu(A[k]).numpy())
A.append(np.dot(Z[p-1], W[p]) + B[p])
Z.append(sigmoid(A[p]))

# check activations/z
pred = keras_model.predict(X_train)
assert np.round(Z[-1].numpy().sum(), 0) == np.round(pred.sum(), 0)

# compute effective weights/bias
W_eff = np.zeros(X_train.shape)
for en, x in enumerate(X_train):
    w_temp = W[0]
    for k in range(p):
        w_temp = np.dot(w_temp * (A[k][en]>0), W[k+1])
    W_eff[en, :] = w_temp.T[0]
    
B_eff = np.zeros(X_train.shape[0])
for en, x in enumerate(X_train):
    for k in range(0, p):
        b_temp = B[k]
        for h in range(k, p):
            b_temp = np.dot(b_temp * (A[h][en]>0), W[h+1])
        B_eff[en] = B_eff[en] + copy.deepcopy(b_temp)[0]
B_eff = B_eff + B[p][0]

# check effective weights
zz = np.zeros(X_train.shape[0])
for g in range(X_train.shape[0]):
    zz[g] = np.dot(X_train[g], W_eff[g]) + B_eff[g]
import math
def sigmoid_(x):
  return 1 / (1 + math.exp(-x))
pred_effective = [sigmoid_(zz_) for zz_ in zz]
assert np.round(Z[-1].numpy().sum(), 0) == np.round(np.sum(pred_effective), 0)

# create clusters based on Z
mat = Z[0]
for z in range(1, p):
    mat = np.hstack((mat, Z[z]))
mat = mat>0
clu = mat.dot(1 << np.arange(mat.shape[-1]))


df_res = pd.DataFrame()
# fig, axes = pl.subplots(2, 4)

df_metaresults = pd.DataFrame()
for en, kk in enumerate(np.unique(clu)):
    explaination_ = W_eff[clu==kk, :] #####+ np.expand_dims(B_eff[clu==kk], 1)
    pred_cluster = 1*(keras_model.predict(X_train)[clu==kk]>0.5)
    positive = np.round(np.sum(pred_cluster)/pred_cluster.shape[0], 2)
    df_res['Feature'] = [nam for nam in feature_names]#p.arange(1, n_features+1)
    df_res['Feature Importance'] = np.mean(explaination_, 0)
    df_res['Abs score'] = np.abs(df_res['Feature Importance'])
    df_plot = df_res.sort_values(['Abs score'], ascending=False)
    seq = "{0:b}".format(kk).zfill(n_neurons)[::-1]
    df_metaresults.loc[seq, 'Support'] = pred_cluster.shape[0]/(X_train.shape[0])#np.round(pred_cluster.shape[0]/(n_samples*train_size),2)
    df_metaresults.loc[seq, 'Positive'] = positive
    df_metaresults.loc[seq, 'Avg instance'] = str(list( np.round((X_train)[clu==kk].mean(0), 2) )) #np.round(
    df_metaresults.loc[seq, feature_names] = df_res['Feature Importance'].values
    df_plot.iloc[:100].plot.bar(fontsize=12, x='Feature', y='Feature Importance', rot=90, legend=False)
    pl.ylabel('Feature Importance')
    pl.xlabel('')
    # female_df = df_task.loc[train_index]['gender'][clu==kk]
    # female = np.round(np.sum(female_df)/female_df.shape[0], 2)
    # df_metaresults.loc[seq, 'Male'] = 1-female

print(df_metaresults.round(2))



Y_pred = 1*(keras_model.predict(X_test)>0.5)[:, 0]
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


