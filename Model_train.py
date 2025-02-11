import numpy as np
import pandas as pd
import pickle
df=pd.read_csv('car_price_prediction.csv')
df.drop_duplicates(subset=['ID'],inplace=True)
df['Mileage']=df['Mileage'].str.replace('km','').astype(int)
df['Leather interior']=df['Leather interior'].map({'Yes':1,'No':0}).astype(float)
df['Levy']=df['Levy'].replace('-','0').astype(int)
df=df.drop('ID',axis=1)

new_df=df
from types import LambdaType
new_df['Turbo']=new_df['Engine volume'].apply( lambda x:1 if 'Turbo' in x else 0)
new_df['Engine volume']=new_df['Engine volume'].str.replace('Turbo','').astype(float)

new_df['Wheel']=new_df['Wheel'].map({'Left wheel':1,'Right-hand drive':0}).astype(float)
new_df=new_df.drop(['Leather interior'] ,axis=1)
df_setect_out=df[['Levy','Engine volume','Mileage','Cylinders','Airbags']]

outlier_cols = []

for column in df_setect_out.columns:

    Q1 = df_setect_out[column].quantile(0.25)
    Q3 = df_setect_out[column].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers based on the IQR
    outliers = (df_setect_out[column] < Q1 - 1.5 * IQR) | (df_setect_out[column] > Q3 + 1.5 * IQR)
    if any(outliers):
        outlier_cols.append(column)

print("Columns with outliers:", outlier_cols)
data = {
    'Levy': [100, 150, 200, 250, 300, 5000],
    'Engine volume': [1.5, 2.0, 2.5, 3.0, 4.0, 10.0],
    'Mileage': [5000, 10000, 15000, 20000, 25000, 500000],
    'Cylinders': [4, 6, 8, 12, 16, 32]
}

df_select_out = pd.DataFrame(data)

# Function to replace outliers in a column with a specific value
def replace_outliers(column, replace_value):
   Q1 = column.quantile(0.25)
   Q3 = column.quantile(0.75)
   IQR = Q3 - Q1

   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR

   column.loc[(column < lower_bound) | (column > upper_bound)] = replace_value

for col in ['Levy', 'Engine volume', 'Mileage', 'Cylinders']:
    replace_outliers(df_select_out[col], replace_value=df_select_out[col].median())
num_data=new_df.select_dtypes(include=['int64','float64'])
cat_data=new_df.select_dtypes(include=['object'])
num_df=num_data
cat_df=cat_data
y=new_df['Price']
num_df.drop(['Price'],axis=1,inplace=True)
cat_df.drop(['Model'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
num_cols = num_df.columns
cat_cols = cat_df.columns
num_pipline=Pipeline(steps=[('scaler',StandardScaler())])
full_pipline=ColumnTransformer([('num',num_pipline,num_cols),
                             ('cat',OneHotEncoder(),cat_cols)])
new_data=full_pipline.fit_transform(new_df)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(new_data,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(random_state=123)
model.fit(x_train,y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('pipeline.pkl', 'wb') as f:
    pickle.dump(full_pipline, f)

print("Models training complete and saved!")



