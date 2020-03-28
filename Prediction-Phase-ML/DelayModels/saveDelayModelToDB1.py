import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from imblearn.over_sampling import SMOTE

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
#from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
import pickle
import time
import pymongo

traindata1 = pd.read_csv('DelayData2016.csv')
traindata2 = pd.read_csv('DelayData2017.csv')
testdata = pd.read_csv('DelayData2018.csv')
birdStrike = pd.read_csv('BirdStrikes.csv')
traindata2.UNIQUE_CARRIER.unique()
list(traindata1)
traindata1 = traindata1[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]

traindata2 = traindata2[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]


testdata = testdata[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','FL_DATE','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','ARR_DELAY','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches']]
testdata = testdata.dropna()
aggr_dataset = [traindata1 , traindata2 , testdata]
data = pd.concat(aggr_dataset)
df = data
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','ARR_DELAY','FL_DATE']]

df = pd.merge(df,birdStrike,on=['ORIGIN'])
df = df[['YEAR','MONTH','DAY_OF_MONTH','FL_NUM',
'DAY_OF_WEEK','UNIQUE_CARRIER','ORIGIN','DEST','DEP_HOUR','ARR_HOUR','CRS_ELAPSED_TIME',
'DISTANCE','traffic','O_SurfaceTemperatureFahrenheit','O_CloudCoveragePercent','O_WindSpeedMph','O_PrecipitationPreviousHourInches',
'O_SnowfallInches','D_SurfaceTemperatureFahrenheit','D_CloudCoveragePercent','D_WindSpeedMph','D_PrecipitationPreviousHourInches','D_SnowfallInches','Bird_Strike',
'ARR_DELAY','FL_DATE']]

df.dtypes
df = df.drop_duplicates()
df['ARR_DELAY'] = df['ARR_DELAY'].clip_lower(0)
cols_to_transform = ['UNIQUE_CARRIER','ORIGIN','DEST']
carrierlist = traindata1.UNIQUE_CARRIER.unique()
df = df[df.UNIQUE_CARRIER.isin(carrierlist)]


df = pd.get_dummies(df, columns = cols_to_transform )
#df_binary = df[cols_to_transform]
#encoder = ce.OneHotEncoder(cols=cols_to_transform)
#data_encoder = encoder.fit_transform(df_binary)
#df_e = df.drop(columns=cols_to_transform)
#data_final = pd.concat([df_e, data_encoder], axis=1)
#data_final = data_final.drop(columns=['intercept'])
#df = data_final 
df = df.dropna()



df1 = df[df['YEAR']==2016]
df1 = df1.drop(['YEAR','FL_DATE'], axis=1)
df2 = df[df['YEAR']==2017]
df2 = df2.drop(['YEAR','FL_DATE'], axis=1)
df3 = df[df['YEAR']==2018]
df3 = df3.drop(columns=['YEAR'])




aggr_dataset = [df1,df2]
traindata = pd.concat(aggr_dataset)
traindata = traindata.dropna()


z = np.abs(stats.zscore(traindata))
z= pd.DataFrame(z)
z=z.fillna(0)
z=z.values
traindata = traindata[(z < 3.5).all(axis=1)]


traindata['DELAY_CLASS'] = 3
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=0)&(traindata.ARR_DELAY <80)  ,0,3)
data1 = traindata[traindata['DELAY_CLASS']==0]
traindata['DELAY_CLASS'] = np.where((traindata.ARR_DELAY>=80)&(traindata.ARR_DELAY >0)  ,1,3)
data2 = traindata[traindata['DELAY_CLASS']==1]

aggr_dataset = [data1,data2]
dataModel = pd.concat(aggr_dataset)
train_labels = dataModel['DELAY_CLASS']
traindata = dataModel.drop(columns=['DELAY_CLASS'])
list(traindata)
sm = SMOTE(random_state=1)
X_train_res, y_train_res = sm.fit_sample(traindata, train_labels)

X_train_res = pd.DataFrame(X_train_res)
X_train_res = X_train_res.rename(columns = {0:'MONTH',1:'DAY_OF_MONTH',2:'FL_NUM',3:'DAY_OF_WEEK',4:'Dep_Hour',
5:'Arr_Hour', 6:'CRS_ELAPSED_TIME', 7:'DISTANCE',
8:'traffic',9:'O_SurfaceTemperatureFahrenheit',10:'O_CloudCoveragePercent',
11:'O_WindSpeedMph',12:'O_PrecipitationPreviousHourInches',13:'O_SnowfallInches',
14:'D_SurfaceTemperatureFahrenheit',15:'D_CloudCoveragePercent',16:'D_WindSpeedMph',
17:'D_PrecipitationPreviousHourInches',18:'D_SnowfallInches',19:'Bird_Strike',20:'ARR_DELAY',
21:'UNIQUE_CARRIER_AA',
22:'UNIQUE_CARRIER_B6',23:'UNIQUE_CARRIER_DL',24:'UNIQUE_CARRIER_EV',
25:'UNIQUE_CARRIER_F9',26:'UNIQUE_CARRIER_NK',
27:'UNIQUE_CARRIER_OO',28:'UNIQUE_CARRIER_UA',
29:'UNIQUE_CARRIER_VX',30:'UNIQUE_CARRIER_WN',31:'ORIGIN_BOS',32:'ORIGIN_DEN',
33:'ORIGIN_DFW',34:'ORIGIN_EWR',35:'ORIGIN_IAH',36:'ORIGIN_JFK',
37:'ORIGIN_LGA',38:'ORIGIN_ORD',39:'ORIGIN_PHL',40:'ORIGIN_SFO',41:'DEST_BOS',
42:'DEST_DEN',43:'DEST_DFW',44:'DEST_EWR',45:'DEST_IAH',46:'DEST_JFK',
47:'DEST_LGA',48:'DEST_ORD',49:'DEST_PHL',50:'DEST_SFO'})

y_train_res = X_train_res['ARR_DELAY']
X_train_res = X_train_res.drop(columns=['ARR_DELAY'],axis=1)
df3 = df3.dropna()
X_test = df3
X_test_for_delay = X_test.copy()
y_test = X_test['ARR_DELAY']
X_test = X_test.drop(columns=['ARR_DELAY'],axis=1)

y_train_res = pd.Series(y_train_res)
list(X_train_res)


coll = ['CRS_ELAPSED_TIME','DISTANCE','O_SurfaceTemperatureFahrenheit',
 'O_CloudCoveragePercent','O_WindSpeedMph','D_SurfaceTemperatureFahrenheit',
 'D_CloudCoveragePercent','D_WindSpeedMph']

scaler = StandardScaler()
scaler.fit(X_train_res[coll])
X_train_res[coll] = scaler.transform(X_train_res[coll])
X_test[coll] = scaler.transform(X_test[coll])


poly = PolynomialFeatures(2)
poly.fit(X_train_res)
X_train_res = poly.transform(X_train_res)
# X_test = poly.transform(X_test)

print(X_train_res.shape)



clf = linear_model.Ridge(alpha=1.2)
print(clf)
clf.fit(X_train_res, y_train_res)


def save_model_to_db(model, client, db, dbconnection, model_name):
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time':time.time()})
    print(info.inserted_id, ' saved with this id successfully!')
    
    details = {
        'inserted_id':info.inserted_id,
        'model_name':model_name,
        'created_time':time.time()
    }
    
    return details

details = save_model_to_db(model = clf, client ='mongodb://localhost:27017/', db = 'models', 
                 dbconnection = 'regression', model_name = 'delay_1')