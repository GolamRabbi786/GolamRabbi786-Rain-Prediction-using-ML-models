# Importing Libraries
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
%matplotlib inline
df=  pd.read_csv("../input/weather-dataset/weatherAUS.csv")
pd.set_option("display.max_columns", None)
df
df.nunique()

num_var = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_var = [feature for feature in num_var if len(df[feature].unique()) <= 25]
cont_var = [feature for feature in num_var if feature not in discrete_var]
categ_var = [feature for feature in df.columns if feature not in num_var]
df[categ_var]
df.isnull().sum()
df.isnull().sum()*100/len(df)
def find_var_type(var):
    
    if var in discrete_var:
        print("{} is a Numerical Variable, Discrete in nature".format(var))
    elif var in cont_var :
        print("{} is a Numerical Variable, Continuous in nature".format(var))
    else :
        print("{} is a Categorical Variable".format(var))
find_var_type('Cloud3pm')

def RandomSampleImputation(df, feature):
    df[feature]=df[feature]
    random_sample=df[feature].dropna().sample(df[feature].isnull().sum(),random_state=0)
    random_sample.index=df[df[feature].isnull()].index
    df.loc[df[feature].isnull(),feature]=random_sample
RandomSampleImputation(df, "Cloud9am")
RandomSampleImputation(df, "Cloud3pm")
RandomSampleImputation(df, "Evaporation")
RandomSampleImputation(df, "Sunshine")

df.isnull().sum()*100/len(df)

find_var_type('RainToday')

def MeanImputation(df, feature):
    df[feature]= df[feature]
    mean= df[feature].mean()
    df[feature]= df[feature].fillna(mean)
    
MeanImputation(df,'Pressure3pm')
MeanImputation(df, 'Pressure9am')
MeanImputation(df, 'MinTemp')
MeanImputation(df, 'MaxTemp')
MeanImputation(df, 'Rainfall')
MeanImputation(df, 'WindGustSpeed')
MeanImputation(df, 'WindSpeed9am')
MeanImputation(df, 'WindSpeed3pm')
MeanImputation(df, 'Pressure9am')
MeanImputation(df, 'Humidity9am')
MeanImputation(df, 'Humidity3pm')
MeanImputation(df, 'Temp3pm')
MeanImputation(df, 'Temp9am')

df.isnull().sum()*100/len(df)

# Plotting a HeatMap

corrmat = df.corr(method = "spearman")
plt.figure(figsize=(20,20))

#plot heat map
g=sns.heatmap(corrmat,annot=True)
Analysis for Continuous variables

for feature in cont_var:
    data=df.copy()
    sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.figure(figsize=(15,15))
    plt.show()

for feature in cont_var:
    data=df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))

#One Hot Encoding

df["RainToday"] = pd.get_dummies(df["RainToday"], drop_first = True)
df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)
df

#Lable Encoding

for feature in categ_var:
    print(feature, (df.groupby([feature])["RainTomorrow"].mean().sort_values(ascending = False)).index)
windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
       'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
       'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
       'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}

df["WindGustDir"] = df["WindGustDir"].map(windgustdir)
df["WindDir9am"] = df["WindDir9am"].map(winddir9am)
df["WindDir3pm"] = df["WindDir3pm"].map(winddir3pm)
df["WindGustDir"].value_counts().index
df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])
df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])

df.isnull().sum()*100/len(df)

df.head()
df_loc = df.groupby(["Location"])["RainTomorrow"].value_counts().sort_values().unstack()
df_loc.head()
df_loc[1].sort_values(ascending = False)
df_loc[1].sort_values(ascending = False).index

mapped_location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}
df["Location"] = df["Location"].map(mapped_location)

# Mapping Data

df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")
df["Date_month"] = df["Date"].dt.month
df["Date_day"] = df["Date"].dt.day
sns.countplot(df["RainTomorrow"])
df= df.drop(['Date'],axis=1)
df.head()

# Plotting Q-Q Plot

import scipy.stats as stats
import pylab
def plot_curve(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stats.probplot(df[feature],dist='norm',plot=pylab)
    plt.title(feature)
    plt.show()
for i in cont_var:
    plot_curve(df, i)
    
# Splitting the data

x = df.drop(["RainTomorrow"], axis=1)
y = df["RainTomorrow"]

from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(x)
X= scale.transform(x)
x.columns
X=pd.DataFrame(X,columns=x.columns)
X.head()
y.head()

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 0)

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
ranfor= RandomForestClassifier()
ranfor.fit(X_train,y_train)
ypred= ranfor.predict(X_test)
print(confusion_matrix(y_test,ypred))
print(accuracy_score(y_test,ypred))
print(classification_report(y_test,ypred))
metrics.plot_roc_curve(ranfor, X_test, y_test)
metrics.roc_auc_score(y_test, ypred, average=None) 

# Gaussian NB
from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb.fit(X_train,y_train)
ypred2= gnb.predict(X_test)
print(confusion_matrix(y_test,ypred2))
print(accuracy_score(y_test,ypred2))
print(classification_report(y_test,ypred2))
metrics.plot_roc_curve(gnb, X_test, y_test)
metrics.roc_auc_score(y_test, ypred2, average=None) 

# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
ypred3= knn.predict(X_test)
print(confusion_matrix(y_test,ypred3))
print(accuracy_score(y_test,ypred3))
print(classification_report(y_test,ypred3))
metrics.plot_roc_curve(knn, X_test, y_test)
metrics.roc_auc_score(y_test, ypred3, average=None) 

# XGB Classifier
from xgboost import XGBClassifier
xgb= XGBClassifier()
xgb.fit(X_train,y_train)
ypred4= xgb.predict(X_test)
print(confusion_matrix(y_test,ypred4))
print(accuracy_score(y_test,ypred4))
print(classification_report(y_test,ypred4))
metrics.plot_roc_curve(xgb, X_test, y_test)
metrics.roc_auc_score(y_test, ypred4, average=None)
# We will save the best performing model i.e. XGB Classsifier model in our pickle file

import pickle
file = open('rain_XGBnew_model.pkl', 'wb')
pickle.dump(xgb, file)

# model = pickle.load(open(&quot;rain_XGBnew_model.pkl&quot;, &quot;rb&quot;))
# Load the model
with open('rain_XGBnew_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sample input data 
sample_data = {
    'MinTemp': 20.0,
    'MaxTemp': 25.0,
    'Rainfall': 5.0,
    'WindGustSpeed': 30.0,
    'WindSpeed9am': 15.0,
    'WindSpeed3pm': 20.0,
    'Humidity9am': 80.0,
    'Humidity3pm': 70.0,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1010.0,
    'Temp9am': 22.0,
    'Temp3pm': 24.0,
    'WindGustDir': 1,
    'WindDir9am': 2,
    'WindDir3pm': 3,
    'RainToday': 1,
    'Location': 10,
    'Date_month': 1, 
    'Date_day': 1,
    'Cloud3pm': 5.0,  
    'Evaporation': 4.0,
    'Sunshine': 7.0,
    'Cloud9am': 4.0,
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame([sample_data])

# Make predictions
predictions = model.predict(sample_df)

# Display predictions
if predictions[0] == 1:
    print("There will be rain Tomorrow.")
else:
    print("There will be no rain Tomorrow.")