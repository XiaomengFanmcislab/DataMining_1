import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_2\winemag-data-130k-v2.csv'
txt=open(r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_2\Wine_reviews_result1.txt','w',encoding = 'utf-8')
problem_1 = 1
file = pd.read_csv(data_name)
data = pd.DataFrame(file)




def fiveNumber(nums):
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.nanpercentile(nums,25)
    Median=np.nanmedian(nums)
    Q3=np.nanpercentile(nums,75)
    
    
    
    return Minimum,Q1,Median,Q3,Maximum

def lost_edition1():
  data_lost1 = data.copy()
  data_lost1 = data_lost1.dropna()
  return data_lost1  

def lost_edition2(price_np):
  
  data_lost2 = data.copy()
  price_mode = mode(price_np)
  
  mode_temp = price_mode[0].tolist()[0]
  data_lost2['price'] = data_lost2['price'].fillna(mode_temp)
  return data_lost2


# 使用随机森林的方法来拟合价值数据
def set_missing_prices():

    data_lost3 = data.copy()

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    price_df = data_lost3[['price', 'points']]

    # 乘客分成已知年龄和未知年龄两部分
    known_price = price_df[price_df.price.notnull()].values
    unknown_price = price_df[price_df.price.isnull()].values

    # y即目标年龄
    y = known_price[:, 0]

    # X即特征属性值
    X = known_price[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedprice = rfr.predict(unknown_price[:, 1:])
#     print predictedAges
    # 用得到的预测结果填补原缺失数据
    data_lost3.loc[ (data_lost3.price.isnull()), 'price' ] = predictedprice

    return data_lost3

def knn_missing_filled(k = 3, dispersed = True):

    data_lost4 = data.copy()
    x_train = data_lost4[data_lost4.price.notnull()]['points'].values.reshape(-1,1)
    y_train = data_lost4[data_lost4.price.notnull()]['price'].values.reshape(-1,1)
    print(len(x_train))
    print(len(y_train))
    test = data_lost4[data_lost4.price.isnull()]['points'].values.reshape(-1,1)

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    data_lost4.loc[ (data_lost4.price.isnull()), 'price' ] = clf.predict(test)

    return data_lost4



print(file.columns)
atts = file.columns
num=0
if problem_1:
  for i in atts[1:]:
    print(i)
    if i!= 'points' and i!= 'price' and i!='description':
        dict_temp = {}
        for j in file.loc[:,i]:
            if j in dict_temp.keys():
                dict_temp[j]+=1
            else:
                dict_temp[j]=1
                num+=1
    
        # print(dict_temp)  
        txt.write(i+'\n')
        txt.write(str(dict_temp))
        txt.write('\n')
        print(num)
  txt.close()
 

price_num_list = [] 
att_pool = ['price']
for att in att_pool:
  
  for point in file.loc[:,att]:
   if isinstance(point,int) or isinstance(point,float):
    price_num_list.append(point)
   else:
    print(point)

price_np = np.array(price_num_list)
min_temp, Q1, median, Q3, max_temp, lower_limit,upper_limit = fiveNumber(price_num_list)
print(att+' five number:')
print(min,Q1,median,Q3,max,lower_limit,upper_limit)

points_num_list = []
att_pool = ['points']
for att in att_pool:
  
  for point in file.loc[:,att]:
   if isinstance(point,int) or isinstance(point,float):
    points_num_list.append(point)
   else:
    print(point)
points_np = np.array(points_num_list)

min_temp, Q1, median, Q3, max_temp = fiveNumber(points_num_list)
print(att+' five number:')
print(min,Q1,median,Q3,max)

print(data.isnull().sum())



data_lost1 = lost_edition1()
data_lost1.to_csv('data_delete.csv')
print(data_lost1.isnull().sum())

data_lost2 = lost_edition2(price_np)
data_lost2.to_csv('data_mode.csv')
print(data_lost2.isnull().sum())

data_lost3 = set_missing_prices()
data_lost3.to_csv('data_forest.csv')
print(data_lost3.isnull().sum())

print('---------------------------------------------')
data_lost4 = knn_missing_filled()
data_lost4.to_csv('data_knn.csv')
print(data_lost4.isnull().sum())


