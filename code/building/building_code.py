import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn.ensemble
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\building\building-violations.csv'

txt=open(r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\building\building-violations_result1.txt','w',encoding = 'utf-8')
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
    
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR #下限值
    upper_limit=Q3+1.5*IQR #上限值
    
    return Minimum,Q1,Median,Q3,Maximum


####删去所有有空缺值的item
def lost_edition1():
    data_lost1 = data.copy()
    data_lost1 = data_lost1.dropna()
    return data_lost1  



def lost_edition2(att_list):
  
    data_lost2 = data.copy()


    for i in att_list:
        i_temp_np = data_lost2[data_lost2[i] != np.NaN].loc[:,i].values
        i_temp_mode = mode(i_temp_np)[0].tolist()[0]
        data_lost2[i] = data_lost2[i].fillna(i_temp_mode)
    
    return data_lost2




# 使用随机森林的方法来拟合价值数据
def set_missing_prices(data_temp, att_list, target):

    att_list = att_list.copy()
    att_list.insert(0,target)
    print('----------------------')
    print(att_list)

    data_lost3 = data_temp.copy()

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    # 首先根据INSPECTION NUMBER', 'PROPERTY GROUP'填充 'Community Areas'的数据
    Community_df = data_lost3[att_list]
    goal = att_list[0]

    # 乘客分成已知年龄和未知年龄两部分
    temp = Community_df[Community_df[goal].notnull()]
    print(temp.isnull().sum())

    known_Community = Community_df[Community_df[goal].notnull()].values
    unknown_Community = Community_df[Community_df[goal].isnull()].values

    # y即目标值
    y = known_Community[:, 0]

    # X即特征属性值
    X = known_Community[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedprice = rfr.predict(unknown_Community[:, 1:])
#     print predictedAges
    # 用得到的预测结果填补原缺失数据
    data_lost3.loc[ (data_lost3[goal].isnull()), goal ] = predictedprice


    return data_lost3


def lost_edition3(complete_list, uncomplete_list ):
    data_lost3 = data.copy()

    for i in uncomplete_list:

        
        print(complete_list)
        data_lost3 = set_missing_prices(data_lost3, complete_list, i)
        print(data_lost3.isnull().sum())

    return data_lost3





def knn_missing_filled(target, complete_list, data, k = 3, dispersed = True):


    # temp = x_train = data_temp[data_temp[target].notnull()][complete_list]
    # print(temp.isnull().sum())
    # print('-----------------------------------------------------')
    # y_temp = data_temp[data_temp[target].notnull()][target]
    # complete_list.insert(0, target)
    # print(y_temp.isnull().sum())
    data_temp = data.copy()
    print('----------------------------------')
    # print(target)
    # print(att_list)
    complete_list_temp = complete_list.copy()
    complete_list_temp.insert(0, target)
    print(complete_list_temp)

    data_temp_temp = data_temp[complete_list_temp]

    x_train = data_temp_temp[data_temp_temp[target].notnull()].values[:,1:]

    y_train = data_temp_temp[data_temp_temp[target].notnull()].values[:,0].reshape(-1,1)
     
    test = data_temp_temp[data_temp_temp[target].isnull()].values[:,1:]

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    data_temp.loc[ (data_temp[target].isnull()), target ] = clf.predict(test)

    return data_temp


def lost_edition4(complete_list, uncomplete_list):

    data_lost4 = data.copy()

    for i in uncomplete_list:
        print('**********************')
        print(i)
        print(complete_list)
        # complete_list.insert(0, i)
        data_lost4 = knn_missing_filled(i, complete_list, data_lost4)
        print(data_lost4.isnull().sum())

    return data_lost4





print(file.columns)
atts = file.columns
num=0
att_num = ['INSPECTION NUMBER', 'PROPERTY GROUP', 'Community Areas', 'Census Tracts','Wards','Historical Wards 2003-2015']
complete_list = ['INSPECTION NUMBER', 'PROPERTY GROUP']
uncomplete_list = ['Community Areas', 'Census Tracts','Wards','Historical Wards 2003-2015']
att_nominal = [ 'VIOLATION LOCATION', 'VIOLATION ORDINANCE', 'INSPECTOR ID', 'INSPECTION STATUS', 'INSPECTION CATEGORY', 'DEPARTMENT BUREAU', 'STREET DIRECTION', 'STREET NAME', 'STREET TYPE']

if problem_1:
  for i in att_nominal :
    print(i)
    
    dict_temp = {}
    for j in data.loc[:,i]:
        if j in dict_temp.keys():
            dict_temp[j]+=1
        else:
            dict_temp[j]=1
            num+=1
    
        # print(dict_temp)  
        txt.write(i+'\n')
        txt.write(str(dict_temp))
        txt.write('\n')
    break
        # print(num)
  txt.close()
 


for i in att_num:
    data_i = data.copy()
    data_i_np = data_i.dropna(subset =[i] ).loc[:,i].values
    min_temp, Q1, median, Q3, max_temp = fiveNumber(data_i_np)
    print(i+' five number:')
    print(min_temp,Q1,median,Q3,max_temp)


print(data.isnull().sum())



data_lost1 = lost_edition1()
data_lost1.to_csv('data_delete.csv')
print(data_lost1.isnull().sum())

data_lost2 = lost_edition2(att_num)
data_lost2.to_csv('data_mode.csv')
print(data_lost2.isnull().sum())

data_lost3 = lost_edition3(complete_list, uncomplete_list)
data_lost3.to_csv('data_forest.csv')
print(data_lost3.isnull().sum())

print('---------------------------------------------')
data_lost4 = lost_edition4(complete_list, uncomplete_list)
data_lost4.to_csv('data_knn_3.csv')
print(data_lost4.isnull().sum())


