import pandas as pd
import numpy as np
data_name = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\winemag-data_first150k.csv'
data_name2 = r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\winemag-data-130k.csv'
txt=open(r'C:\Users\Xiaomeng Fan\Desktop\研究生的各种作业\数据挖掘\作业四\Wine_reviews_result1.txt','w',encoding = 'utf-8')
problem_1 = 0
def fiveNumber(nums):
    #五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum=min(nums)
    Maximum=max(nums)
    Q1=np.nanpercentile(nums,25)
    Median=np.nanmedian(nums)
    Q3=np.nanpercentile(nums,75)
    

    return Minimum,Q1,Median,Q3,Maximum

file = pd.read_csv(data_name)
data = pd.DataFrame(file)


print(file.columns)
atts = file.columns
num=0
if problem_1:
  for i in atts[1:]:
    print(i)
    if i!= 'points' and i!= 'price' and i!='description':
        dict = {}
        for j in file.loc[:,i]:
            if j in dict.keys():
                dict[j]+=1
            else:
                dict[j]=1
                num+=1
    
            
        txt.write(i+'\n')
        txt.write(str(dict))
        txt.write('\n')
        print(num)
  txt.close()
  
att_pool = ['price']
for att in att_pool:
  num_list = []
  for point in file.loc[:,att]:
   if isinstance(point,int) or isinstance(point,float):
    num_list.append(point)
   else:
    print(point)
  point_np = np.array(num_list)
min_temp, Q1, median, Q3, max_temp, lower_limit,upper_limit = fiveNumber(num_list)
print(att+' five number:')
print(min,Q1,median,Q3,max,lower_limit,upper_limit)

att_pool = ['points']
for att in att_pool:
  num_list = []
  for point in file.loc[:,att]:
   if isinstance(point,int) or isinstance(point,float):
    num_list.append(point)
   else:
    print(point)
  point_np = np.array(num_list)

min_temp, Q1, median, Q3, max_temp, lower_limit, upper_limit = fiveNumber(num_list)
print(att+' five number:')
print(min,Q1,median,Q3,max,lower_limit,upper_limit)

print(data.isnull().sum())




