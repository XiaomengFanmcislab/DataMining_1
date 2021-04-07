import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\winemag-data_first150k.csv'
data_delete_name  = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\data_delete.csv'
data_mode_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\data_mode.csv'
data_forest_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\data_forest.csv'
data_knn_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\wine_1\data_knn.csv'


file = pd.read_csv(data_name)
data = pd.DataFrame(file)

file_delete = pd.read_csv(data_delete_name)
data_delete = pd.DataFrame(file_delete)

file_mode = pd.read_csv(data_mode_name)
data_mode = pd.DataFrame(file_mode)

file_forest = pd.read_csv(data_forest_name)
data_forest = pd.DataFrame(file_forest)

file_knn = pd.read_csv(data_knn_name)
data_knn = pd.DataFrame(file_knn)

def show_boxplot(data_temp, att_temp):

    data_temp1 = data_temp.copy()

    for i in att_temp:
        data_i = data_temp1.dropna(subset=[i])
        att_i = data_i.loc[:,i].values
        plt.boxplot(att_i, notch=False, sym='o', vert=True)
        plt.title(i)
        plt.show()
        

def show_hist(data_temp, att_temp, bin_numbers):
    data_temp1 = data_temp.copy()

    for i in att_temp:
        data_i = data_temp1.dropna(subset=[i])
        att_i = data_i.loc[:,i].values
        plt.hist(att_i, bins = bin_numbers, color = '#607c8e', rwidth=5)
        plt.xlabel(i)
        plt.ylabel('frequency')
        plt.title('Histogram of'+ i )
        plt.show()


def subplot_show(data1, att1, data2, att2, bin_num, title1, title2):

    data1_temp = data1.copy()

    data_i = data1_temp.dropna(subset=[att1])
    data1_att = data_i.loc[:,att1].values
    data2_att = data2.loc[:,att2].values

    plt.subplot(221)
    plt.hist(data1_att,bins = bin_num, color = '#607c8e', rwidth=5)
    plt.xlabel(att1)
    plt.ylabel('frequency')
    plt.title('Histogram of '+ title1 )

    plt.subplot(222)
    plt.hist(data2_att,bins = bin_num, color = '#607c8e', rwidth=5)
    plt.xlabel(att2)
    plt.ylabel('frequency')
    plt.title('Histogram of '+ title2 )

    plt.subplot(223)
    plt.boxplot(data1_att, notch=False, sym='o', vert=True)
    
    plt.title('Box figure of '+title1)

    plt.subplot(224)
    plt.boxplot(data2_att, notch=False, sym='o', vert=True)
    plt.title('Box figure of '+title2)

    plt.show()

    



att = 'points'
# show_hist(data, att)

subplot_show(data, att, data_delete, att, 15, 'raw data', 'processed data')
