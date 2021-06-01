import pandas as pd
import os
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import KFold


os.getcwd()
os.chdir('Documents\Analytics\Fine Windy Day Hackerearth')

#reading the dataset, -99,999,-999,-30 have been treated as missing values
train=pd.read_csv('Data/train.csv',na_values=[-99,999,-999,-30])
test=pd.read_csv('Data/test.csv',na_values=[-99,999,-999,-30])
train['type']='train'
test['type']='test'

#concatenating the train and test data for feature engineering
traindf=pd.concat([train,test])

#generating the features from datetime column
traindf['datetime']=pd.to_datetime(traindf['datetime'])
traindf['month']=traindf['datetime'].dt.month.astype('str')
traindf['week']=traindf['datetime'].dt.week.astype('str')
traindf['dayofMonth']=traindf['datetime'].dt.day.astype('str')
traindf['weekday']=traindf['datetime'].dt.weekday.astype('str')
traindf['hourofDay']=traindf['datetime'].dt.hour.astype('str')
traindf['dayofYear']=traindf.datetime.dt.dayofyear.astype('str')

#getting range columnwise to remove outliers
num_cols_init=traindf.columns[traindf.dtypes!='object'].drop(['windmill_generated_power(kW/h)','datetime'])
df1=train[num_cols_init].mean()+6*train[num_cols_init].std()
df2=train[num_cols_init].mean()-6*train[num_cols_init].std()

'''
Aggregating the numerical column and getting the count, mean, median, minimum, maximum, standard deviation and skew
grouping them by categorical columns. The datetime deatures generated earlier have also been taken as categorical
'''
cat_cols=traindf.columns[traindf.dtypes=='object'].drop(['tracking_id','type'])
num_cols=traindf.columns[traindf.dtypes!='object'].drop(['windmill_generated_power(kW/h)','datetime'])
traindf[cat_cols]=traindf[cat_cols].fillna('Missing')


for col in cat_cols:
    df=traindf[list(num_cols)+[col]].groupby(col).agg(['count','mean','median','min','max','std','skew']).reset_index()
    df.columns=list(map(lambda x:x[0]+col+'_'+x[1],df.columns))
    df.columns=[col]+[*df.columns[1:]]
    traindf=pd.merge(traindf,df,how='left')
del col,df

#Aggregating by grouping by hourofDay and dayofYear columns generated earlier
df=traindf[list(num_cols)+['hourofDay','dayofYear']].groupby(['hourofDay','dayofYear']).agg(['count','mean','median','min','max','std','skew']).reset_index()
df.columns=list(map(lambda x:'hourandday'+x[0]+'_'+x[1],df.columns))
df.columns=['hourofDay','dayofYear']+[*df.columns[2:]]
traindf=pd.merge(traindf,df,how='left')
del df


drop_cols=['tracking_id','datetime','type','windmill_generated_power(kW/h)']
cat_cols=list(traindf.columns[(traindf.dtypes=='object')])
num_cols=list(set(traindf.columns)-set(cat_cols)-set(drop_cols))
cat_cols=list(set(cat_cols)-set(drop_cols))
del drop_cols


#subsetting the train data from concatenated data, dropping rows from train data where target is missing
train_X=traindf[traindf.type=='train']
train_X=train_X[~train_X['windmill_generated_power(kW/h)'].isnull()]
train_Y=train_X['windmill_generated_power(kW/h)']
train_X=train_X[cat_cols+num_cols]

#defining a custom eval function for catBoost
class r2Score(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        pred=approxes[0]
        score=max(0,100*r2_score(target,pred))
        return score, 0.0

#setting paramaters for catBoost
params_cb={
    'cat_features': cat_cols,
    'random_seed': 123,
    'n_estimators': 3000,
    'colsample_bylevel': 0.703538993864461,
    'depth': 7,
    'learning_rate':0.03757009932812784}

#subsetting the test data from concatenated data
test_X=traindf[traindf.type=='test']
test_X=test_X[cat_cols+num_cols]

'''
not label encoding the categorical columns as it wont take much time for model building, for faster model building times,
one can label/ordinal encode the categorical columns. This wont cause major change in score.
'''
test_X[cat_cols]=test_X[cat_cols].astype('str')
train_X[cat_cols]=train_X[cat_cols].astype('str')


#doing a 10 fold CV to avoid overfiiting, also building models and predicting for each such fold
kf=KFold(n_splits=10,random_state=1234)
cb_scores=[]
pred_cb=[]
for idxT, idxV in kf.split(train_X):
    df_trainY=train_Y.iloc[idxT]
    df_trainX=train_X.iloc[idxT]
    
    #removing the outliers just before building the model for reliable CV scores    
    for col in num_cols_init:
        df_trainY=df_trainY[np.invert(df_trainX[col]>df1[col])]
        df_trainX=df_trainX[np.invert(df_trainX[col]>df1[col])]
        df_trainY=df_trainY[np.invert(df_trainX[col]<df2[col])]
        df_trainX=df_trainX[np.invert(df_trainX[col]<df2[col])]
    
    df_trainX[cat_cols]=df_trainX[cat_cols].astype('str')
    cb=CatBoostRegressor(**params_cb,eval_metric=r2Score(),early_stopping_rounds=50)
    cb.fit(df_trainX, df_trainY,eval_set=(train_X.iloc[idxV],train_Y.iloc[idxV]),plot=False, verbose=200)
    cb_scores.append(cb.get_best_score().get('validation').get('r2Score'))
    pred_cb.append(cb.predict(test_X))

#calculating weights for ensembling based on validation scores (higher the better)
weights=(cb_scores)/np.sum(cb_scores)
print ('The Local CV is {}'.format(np.sum(weights*cb_scores)))

#multipying the predictions by weights to get the final prediction
pred=np.sum(np.multiply(np.transpose(np.array(pred_cb)),weights),1)
submit=pd.DataFrame({'tracking_id':test['tracking_id'],'datetime':test['datetime'],'windmill_generated_power(kW/h)':pred})
submit.to_csv('Submissions/submit.csv',index=False)