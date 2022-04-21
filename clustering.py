import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer,SilhouetteVisualizer
from sklearn.metrics import silhouette_score
from helpers import eda
pd.set_option('display.expand_frame_repr', False)


transactions= eda.excel_file("transactions","Transactions",refresh=False)
transactions["Transaction_Month"] = transactions['Transaction_Date'].dt.month
transactions["Transaction_Day"] = transactions['Transaction_Date'].dt.day
transactions["Transaction_Week_Day"] = transactions['Transaction_Date'].dt.strftime("%A")
cat_cols, num_cols, cat_but_car,typless_cols =eda.col_types(transactions)
na_col,null_high_col_name=eda.desc_statistics(transactions,num_cols,cat_cols,refresh=True)

def create_unique_df(dataframe,date_col,id_col,amount_col):
    dataframe.dropna(inplace=True)
    today_date = dataframe[date_col].max() +dt.timedelta(days=7)
    diff_day_df = dataframe.groupby([id_col,date_col]).size().reset_index(name='count').groupby([id_col]).agg("count").reset_index()[[id_col,"count"]]
    diff_month_df = dataframe.groupby([id_col,'Transaction_Month']).size().reset_index(name='count').groupby([id_col]).agg("count").reset_index()[[id_col,"count"]]
    unique_custom_df = dataframe.groupby(id_col).agg({date_col: lambda x:(today_date - x.max()).days,
                                                      id_col: lambda x: x.count(),
                                                      amount_col: lambda x: x.sum()}).rename(columns={date_col: 'Recency',
                                                                                                      id_col: 'Frequency',
                                                                                                      amount_col: 'Monetary'}).reset_index()
    unique_custom_df=pd.merge(unique_custom_df,dataframe.groupby(id_col).agg({date_col: lambda x: (today_date - x.min()).days}).rename(columns={date_col:'First_Shopping_Day'}).reset_index(),on=id_col)
    unique_custom_df=pd.merge(unique_custom_df,diff_day_df,on=id_col)
    unique_custom_df=pd.merge(unique_custom_df,diff_month_df,on=id_col)
    unique_custom_df.rename(columns={'count_x':'Diff_Day','count_y':'Diff_Month'},inplace=True)
    return unique_custom_df

#There is a new customer definition according to the business rules determined by the companies and
#these new customers will not be included in the segmentation study since their shopping information has not yet matured.
#That's why we identified new customers here and removed them from our master data.
def new_customer_df(dataframe,First_Shopping_Day_Value=45):
    new_customer = dataframe.loc[dataframe["First_Shopping_Day"]<=First_Shopping_Day_Value].copy()
    new_customer["Segment"] ="New_Customer"
    unique_custom_df_ = dataframe.loc[dataframe["First_Shopping_Day"]>First_Shopping_Day_Value].copy()
    return new_customer, unique_custom_df_
new_customer, unique_custom_df_=new_customer_df(create_unique_df(transactions,"Transaction_Date",'Customer_Number','Total_Amount'))
unique_custom_df_.sort_values(by="Frequency",ascending=False).head(10)
unique_custom_df_ = unique_custom_df_.loc[unique_custom_df_["Frequency"]<120].copy()


#We performed a hierarchical grouping process so that the totals in the groups were equal.
#So, for example, we can observe that the money left by ten people is equal to the money left by one person.
def monetary_micro_segment(dataframe,group_value=4):
    numgroups = group_value
    dataframe['csum'] = dataframe['Monetary'].sort_values().cumsum()
    dataframe['Micro_Segment'] =((dataframe['csum']/  (dataframe['csum'].max() /numgroups))+1).astype(int)
    dataframe['Micro_Segment'] =np.where(dataframe['Micro_Segment']>numgroups,numgroups,dataframe['Micro_Segment'])
    micro_segment_df=dataframe.groupby(dataframe['Micro_Segment'])['Monetary'].agg({"count", "sum", "min", "max"})
    return micro_segment_df
monetary_micro_segment(unique_custom_df_)


def scaler(dataframe,col):
    scaler = MinMaxScaler((0,1))
    unique_custom_df_scaler=scaler.fit_transform(dataframe[col])
    return unique_custom_df_scaler

def cluster(dataframe,silhouette=True):
    k_means = KMeans(n_clusters=5,random_state=42).fit(dataframe)
    if silhouette:
        score = silhouette_score(dataframe, k_means.labels_, metric='euclidean')
        print("Silhouette score :",score)
        visualizer = SilhouetteVisualizer(k_means, colors='yellowbrick')
        visualizer.fit(dataframe)
        visualizer.show()
    return k_means


def predict_cluster(dataframe,dataframe_scaler,silhouette=True):
    predict = cluster(dataframe_scaler,silhouette).fit_predict(dataframe_scaler)
    dataframe['Cluster'] = pd.Series(predict, index=dataframe.index)
    return dataframe
predict_cluster(unique_custom_df_,scaler(unique_custom_df_,["Micro_Segment","Recency","Frequency","Diff_Month"]))

#It was calculated how many times more money the clusters, which were determined by the lift calculation and the clustering method, left each other.
#Thus, by observing a hierarchy, segment names that can be given to customer clusters can be determined.
def lift_calc(dataframe,col_Monetary, col_Frequency,col_Segment):
    new_df= pd.DataFrame()
    new_df["Sum_Monetary"]=dataframe[[col_Monetary,col_Segment]].groupby(col_Segment).agg(["sum"])
    new_df["Sum_Cust"]=dataframe[[col_Frequency,col_Segment]].groupby(col_Segment).agg(["count"])
    new_df["Monetary_Rate"] = new_df["Sum_Monetary"] / new_df["Sum_Monetary"].sum()
    new_df["Cust_Rate"] = new_df["Sum_Cust"] / new_df["Sum_Cust"].sum()
    new_df["Lift"] =  new_df["Monetary_Rate"] /new_df["Cust_Rate"]
    lift_df=new_df.sort_values(by="Lift",ascending=False)
    return lift_df
lift_calc(unique_custom_df_,"Monetary","Frequency","Cluster")


def update_cluster(dataframe,cluster_col,update_value=0,value=4):
    dataframe[cluster_col] =np.where(dataframe[cluster_col]==update_value,value,dataframe[cluster_col])
update_cluster(unique_custom_df_,'Cluster')
lift_df=lift_calc(unique_custom_df_,"Monetary","Frequency","Cluster").reset_index()


def segment_name(dataframe,lift_dataframe,id_col):
    lift_dataframe["Segment"] = ["Platin" if i == 0 else 'Gold'
                                     if i == 1 else 'Silver'
                                     if i == 2 else 'Bronze'
                          for i in list(lift_dataframe.index)]
    dataframe = pd.merge(dataframe, lift_dataframe, on=id_col, how='left')
    return dataframe
print(segment_name(unique_custom_df_,lift_df[["Cluster","Lift"]],"Cluster").head())


def diff_df(dataframe_diff,dataframe,key_col):
    key_diff = set(dataframe_diff[key_col]).difference(dataframe[key_col])
    diff_df = dataframe_diff[dataframe_diff[key_col].isin(key_diff)]
    return diff_df
new_df=diff_df(create_unique_df(transactions,"Transaction_Date",'Customer_Number','Total_Amount'),unique_custom_df_,"Customer_Number")
new_df=diff_df(new_df,new_customer,"Customer_Number")


def predict_micro_segment(dataframe,new_df):
    for j in range(len(monetary_micro_segment(dataframe))-1):
        for i in new_df["Monetary"]:
            if (list(monetary_micro_segment(dataframe)["max"])[j] <= i < list(monetary_micro_segment(dataframe)["max"])[j + 1]) or i > monetary_micro_segment(dataframe)["max"].max():
                new_df["Micro_Segment"] = monetary_micro_segment(dataframe).reset_index()["Micro_Segment"][0]
    return new_df
predict_micro_segment(unique_custom_df_,new_df)

predict_cluster(predict_micro_segment(unique_custom_df_,new_df),scaler(new_df,["Micro_Segment","Recency","Frequency","Diff_Month"]),silhouette=False)
update_cluster(predict_micro_segment(unique_custom_df_,new_df),'Cluster')
segment_name(predict_micro_segment(unique_custom_df_,new_df),lift_df[["Cluster","Lift"]],"Cluster")
segment_name(predict_micro_segment(unique_custom_df_,new_df),lift_df[["Cluster","Lift"]],"Cluster").append(new_customer, ignore_index=True)

