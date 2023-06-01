import pandas as pd
import numpy as np
import re
import time
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import streamlit as st
import plotly.express as px
from pathlib import Path
from tqdm import tqdm
placeholder = st.empty()
from os.path import exists
from streamlit_autorefresh import st_autorefresh
#auto refreshing after every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="Refresh_Server")

##Overall Order and items
Main_set = pd.read_csv('orders.csv')
Main_set.rename(columns={'date|userID|itemID|order': 'data'}, inplace=True)
Main_set[['date', 'user_Id', 'item_Id', 'Order']] = Main_set['data'].str.split('|', expand=True)
Main_set.drop('data', axis=1, inplace=True)

Main_set[["year", "month", "day"]] = Main_set['date'].str.split("-", expand=True)



model_num = 1
#Reading the file that stored previous records about model
record_track = 'Accuarcy_record.csv'
file_exists = exists(record_track)
#files that combine current and previous data
storing_df = 'df_order.csv'
if (file_exists):
    record = pd.read_csv(record_track)
    exist_df = record['file_path'][len(record.index) - 1]

    index_number = record['index_No'][len(record.index) - 1]
    index_number = int(index_number) + 1
    #Reading next csv file (new)
    new_df = 'df_order_0' + str(index_number) + ".csv"
    #previous file
    data1 = pd.read_csv(exist_df, sep='\t')
    #new file
    data2 = pd.read_csv(new_df, sep='\t')

    #combining current and previous file
    Order_df = pd.concat([data1, data2], axis=0)

    #saving the file
    Order_df.to_csv(storing_df, index=False)
else:
    index = model_num
    #new file
    new_df = 'df_order_0' + str(index) + ".csv"
    Order_df = pd.read_csv(new_df, sep='\t')
    #storing file
    Order_df.to_csv(storing_df, index=False)
Order_df.rename(columns={'date|userID|itemID|order': 'data'}, inplace=True)
Order_df[['date', 'user_Id','item_Id','Order']] = Order_df['data'].str.split('|', expand=True)
Order_df.drop('data', axis=1, inplace=True)

#Reading items file

item_df =pd.read_csv(r'items.csv', sep='\t')
item_df.rename(columns={'itemID|brand|feature_1|feature_2|feature_3|feature_4|feature_5|categories': 'data'}, inplace=True)
item_df[['item_Id', 'brand','feature_1','feature_2','feature_3','feature_4','feature_5','categories']] = item_df['data'].str.split('|', expand=True)
item_df.drop('data', axis=1, inplace=True)

#droping categories
item_df.drop('categories', axis=1, inplace=True)

#joining both file
Item_Order_df = pd.merge(Order_df,item_df,on ='item_Id')

#turning order into integer
Item_Order_df[["Order"]] = Item_Order_df[["Order"]].astype("int")

#spliting date
Item_Order_df[["year", "month", "day"]] = Item_Order_df['date'].str.split("-", expand=True)

# method to make target attributes
weekList = []
for i in range(len(Item_Order_df)):

    # 1-7
    if re.match(r"\d{4}-\d{2}-([0][0-7])", Item_Order_df.date.iloc[i]) and (Item_Order_df.Order.iloc[i] > 0):

        weekList.append(1)

    # 8-15
    elif re.match(r"\d{4}-\d{2}-([11][0-5]|[0][8-9])", Item_Order_df.date.iloc[i]) and (
            Item_Order_df.Order.iloc[i] > 0):

        weekList.append(2)

    # 16-22
    elif re.match(r"\d{4}-\d{2}-([11][6-9]|[2][0-2])", Item_Order_df.date.iloc[i]) and (
            Item_Order_df.Order.iloc[i] > 0):
        # if(Item_Order_df.iloc[i, 3] > 0):
        weekList.append(3)

    # 23-31
    elif re.match(r"\d{4}-\d{2}-([2][3-9]|3[01])", Item_Order_df.date.iloc[i]) and (Item_Order_df.Order.iloc[i] > 0):
        # if(Item_Order_df.iloc[i, 3] > 0):
        weekList.append(4)
    else:
        weekList.append(0)

Item_Order_df["prediction"] = weekList


#droping order, date

Item_Order_df.drop('Order', axis=1, inplace=True)
Item_Order_df.drop('date', axis=1, inplace=True)
Item_Order_df.drop('year', axis=1, inplace=True)
Item_Order_df.drop('month', axis=1, inplace=True)


#######
np.random.seed(10)
remove_n = int((len(Item_Order_df)/4)*3.5)
drop_indices = np.random.choice(Item_Order_df.index, remove_n, replace=False)
df_subset = Item_Order_df.drop(drop_indices)
#######
#df_subset =  Item_Order_df
x = df_subset.drop('prediction',axis = 1)
y = df_subset['prediction']
norm = MinMaxScaler().fit(x)
x_norm = norm.transform(x)


#SPILITING THE DATA
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size = 0.20, random_state = 1)

# model declaration
model = Sequential()

# 1st HL and IL
model.add(Dense(units = 600, activation = 'relu', name = 'HL_1', input_dim = 9))
model.add(Dropout(0.2)) # helpful to reduce/avoid overfitting

# 2nd HL
model.add(Dense(units = 6, activation = 'relu', name = 'HL_2'))
model.add(Dropout(0.2))


# OL
model.add(Dense(units = 1, activation = 'relu', name = 'OL')) # TV is binary

# model summary
model.summary()


# model checkpoint using callbacks
file = 'Best_Model.hdf5'
checkpoint = ModelCheckpoint(file, monitor = 'val_accuracy', save_best_only = True, mode = 'max', verbose = 2)
callbacks_list = [checkpoint]


# model compilation
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

# model fitting
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 5, batch_size = 1024, callbacks = callbacks_list)

# predicting
y_pred = model.predict(x_test)

for i in range(len(y_pred)):
  if(y_pred[i][0] >4):
    y_pred[i][0] = y_pred[i][0].astype(int)
  else:
    y_pred[i][0] = np.round(y_pred[i][0])





a = (accuracy_score(y_test, y_pred)*100)
p = precision_score(y_test, y_pred, average='micro')
r = recall_score(y_test, y_pred, average='micro')
f = f1_score(y_test, y_pred, average='micro')

record = 'Accuarcy_record.csv'
file_exists = exists(record)
#df_accuracy.loc[len(df_accuracy.index)] = [a, model_num]

if (file_exists):
    df_record = pd.read_csv(record)
    df_record.loc[len(df_record.index)] = [len(df_record.index)+1,a,p,r,f,storing_df,new_df]
    pre_a = df_record['Accuracy'][len(df_record.index) - 2]
    pre_p = df_record['Precision'][len(df_record.index) - 2]
    pre_r = df_record['Recall'][len(df_record.index) - 2]
    pre_f = df_record['f1-score'][len(df_record.index) - 2]
    df_record.to_csv(record, index=False)

else:
    df_record = pd.DataFrame()
    df_record[['index_No', 'Accuracy', 'Precision','Recall', 'f1-score','file_path','previous_path']] = ""
    df_record.loc[len(df_record.index)] =  [1,a,p,r,f,storing_df,new_df]
    pre_a = 0
    pre_p = 0
    pre_r = 0
    pre_f = 0
    df_record.to_csv(record, index=False)


with placeholder.container():
    kpi1, kpi2, kpi3,kpi4 = st.columns(4)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Accuracy ⏳",
        value= round(a,3),
        delta= round(pre_a,3)
    )
    kpi2.metric(
        label="Precision ⏳",
        value=round(p ,3),
        delta= round(pre_p,3)
    )
    kpi3.metric(
        label="Recall_score ⏳",
        value= round(r,3),
        delta=round(pre_r,3)
    )
    kpi4.metric(
        label="f1_score ⏳",
        value= round(f,3),
        delta= round(pre_f,3)
    )

    list1 = []
    list2 = []
    for i in range(len(y_pred)):
        list1.append(y_pred[i][0])
        if (y_pred[i][0] == 1):
            list2.append('1')
        elif (y_pred[i][0] == 2):
            list2.append('2')
        elif (y_pred[i][0] == 3):
            list2.append('3')
        else:
            list2.append('4')

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        with st.container():
            fig = px.line(df_record,x='index_No', y='Accuracy', title='ACCURACY')
            fig.update_layout(width=500, height=400, bargap=0.05)
            st.write(fig)

            #fig2.update_layout(width=500, height=500, bargap=0.10)
           # fig2.update_traces(textposition="bottom right")

        with st.container():
            cm_1 = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm_1 / np.sum(cm_1),labels=dict(x="Actual label", y="Predicted label", color="Productivity"),
                            title='Accuracy Score: {0} %'.format(accuracy_score(y_test, y_pred)),
                            text_auto=True,aspect="auto", color_continuous_scale='RdBu')
            fig.update_layout(width=500, height=400, bargap=0.05)
            st.write(fig)



    # st.write(classification_report(y_test,y_pred))
    # Confusion Matrix
    with col4:
        with st.container():
            fig = px.bar(x=list1, y=list2, title="Prediction Value", text_auto='.2s', color_continuous_scale='RdBu')
            fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            fig.update_layout(width=500, height=400, bargap=0.05)
            st.write(fig)

        with st.container():
             confusion_matrix_best_model = metrics.confusion_matrix(y_test, y_pred)

             fig = px.imshow(confusion_matrix_best_model,labels=dict(x="Actual label", y="Predicted label", color="Productivity"),
                             title='Accuracy Score: {0}'.format(accuracy_score(y_test, y_pred)*100),
                             text_auto=True, aspect="auto",color_continuous_scale='RdBu')
             fig.update_layout(width=500, height=400, bargap=0.05)
             st.write(fig)


Test_df = pd.read_csv('submission.csv',sep='\t')
Test_df.rename(columns={'userID|itemID|prediction': 'data'}, inplace=True)
Test_df[['user_Id','item_Id','prediction']] = Test_df['data'].str.split('|', expand=True)
Test_df.drop('data', axis=1, inplace=True)
Test_df.drop('prediction', axis=1, inplace=True)
# display the dataframe

Test_df = pd.merge(Test_df,item_df,on ='item_Id',how='left')
Test_df["day"] = ""

Train_group2 = Main_set.groupby(['user_Id'])['day'].median().reset_index()


Test_df["day"] = ""

Test_df['user_Id'] = Test_df['user_Id'].astype('int')
Train_group2['user_Id'] = Train_group2['user_Id'].astype(int)
item1 = Test_df['user_Id']
item2 = Test_df['user_Id']

for i in range(len(item1)):
    if(item2[i] in Train_group2['user_Id']):
        filt = Train_group2['user_Id'] == item1[i]
        v = Train_group2.index[filt].tolist()
        value = v[-1]
        Test_df.iloc[i,8] = Train_group2.iloc[value,1]
    else:
        Test_df.iloc[i,8] = 'NaN'

Main_data = pd.concat([df_subset, Test_df], axis=0)

x = Main_data.drop('prediction',axis = 1)
y = Main_data['prediction']

split = round(len(df_subset))
train_set, test_set = Main_data[:split], Main_data[split:]
print('train on %d instances, test on %d instances' % (len(train_set), len(test_set)))

x = train_set.drop('prediction',axis = 1)
y = train_set['prediction']

#Test set
x_test = test_set.drop('prediction',axis = 1)


from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(x).transform(x)


x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)

model.fit(X, y, epochs=10, batch_size = 1024, shuffle=True)
y_pred = model.predict(x_test)
for i in range(len(y_pred)):
  if(y_pred[i][0] >4):
    y_pred[i][0] = y_pred[i][0].astype(int)
  elif(y_pred[i][0] <1):
      y_pred[i][0] = 0
  else:
    y_pred[i][0] = np.round(y_pred[i][0])

Test_df["prediction"] = y_pred




soluton_df = Test_df[["user_Id","prediction"]]

soluton_df.to_csv('soluton_df.csv', index=False)





