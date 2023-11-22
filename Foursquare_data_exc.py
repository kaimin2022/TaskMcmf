import csv
import numpy as np
import pandas as pd
import time
import random
from collections import Counter

"""
Foursquare dataset_TSMC2014_NYC.csv 签到数据集处理
Tokyo 和 New York 随机生成 1w 条数据
将 数据txt文件 转换成 csv文件：只保留2013年时间的数据
选取NYC文件中 Jan 月份的数据 17号一天的数据 共711条数据左右
"""

Nyc_file = 'dataset_TSMC2014_NYC.csv'
Nyc_1_file = 'dataset_TSMC2013_Jan_17_NYC.csv'
Nyc_2_file = 'dataset_TSMC2013_Jan_17_NYC_2.csv'
Nyc_final_file = 'dataset_TSMC2013_Jan_17_NYC_final.csv'

# data1 = []
# with open(Nyc_file, 'r', newline='', encoding='utf-8') as Nyc_1:
#     csv_file = csv.reader(Nyc_1)
#     for row in csv_file:
#         # print(row[0][-26:-23])  ---- Jan
#         # print(row[0][-22:-20])
#         # if row[0][-22:-20] == '17':
#         if row[0][-26:-20] == 'Jan 17':
#             data1.append(row)
#
# # 保存到新的文件中
# with open(Nyc_1_file, 'w', newline='') as Nyc_2:
#     Nyc_final_file = csv.writer(Nyc_2)
#     Nyc_final_file.writerows(data1)

"""
读取Jan-17同一天的csv数据文件 删除没必要的数据列
# """

# list_index = [2,3,6]
# with open(Nyc_1_file,'r',encoding='utf-8') as fr:
#     lines = fr.readlines()
#     with open(Nyc_2_file,'w+',encoding='utf-8', newline='') as fw:
#         for line in lines:
#             temp = line.split('\t')
#             temp = [n for i, n in enumerate(temp) if i not in list_index]
#             print(temp)
#             time = temp[4]
#             time = time[4:30]
#             temp[4] = time
#             print(temp)
#             # 随机生成一个在线时间 online_time
#             online_time = random.randint(600, 2700)
#             temp[1] = online_time
#             print(temp)
#             writer = csv.writer(fw)
#             writer.writerow(temp)

"""
随机生成 10-50 个工人 和 10-50个任务
"""
# 任务
# Nyc = pd.read_csv('dataset_TSMC2013_Jan_17_NYC_2.csv')
# for tnum in range(10,60,10):
#     Nyc_exc = Nyc.sample(n=tnum)
#     Nyc_exc.to_csv('Task_Jan_17_NYC_' + str(tnum) + '\\' + 'task_NYC_' + str(tnum) + '.csv', index=False, header=None)

# for tnum in range(10,60,10):
#     filename = 'Task_Jan_17_NYC_' + str(tnum) + '\\' + 'task_NYC_' + str(tnum) + '.csv'
#     print(filename)
#     Nyc1 = pd.read_csv(filename,header=None)
#     for j in range(len(Nyc1)):
#         Nyc1.at[j, 0] = str(j + 1)
#     Nyc1.to_csv(filename, index=False, header=0)

# 工人
# Nyc = pd.read_csv('dataset_TSMC2013_Jan_17_NYC_2.csv')
# for tnum in range(10,60,10):
#     Nyc_exc = Nyc.sample(n=tnum)
#     filename = 'Worker_Jan_17_NYC_' + str(tnum) + '\\' + 'worker_NYC_' + str(tnum) + '.csv'
#     Nyc1 = pd.read_csv(filename,header=0)
#     # del Nyc1['0']
#     Nyc1.columns = ['poi_id', 'eff_time', 'lat', 'lon', 'times']
#     print(len(Nyc1))
#     for j in range(len(Nyc1)):
#         Nyc1.at[j, 'poi_id']=str(j+1)
#     Nyc1.to_csv(filename, index=False)




"""
Foursquare dataset_TSMC2014_NYC.csv 签到数据集处理
Tokyo 和 New York 随机生成 1w 条数据
将 数据txt文件 转换成 csv文件：只保留2013年时间的数据
选取Tokyo 文件中 Jan 月份的数据 26号一天的数据 共2500+条数据左右
"""

Tokyo_file = 'dataset_TSMC2014_TKY.csv'
Tokyo_1_file = 'dataset_TSMC2013_Jan_26_Tokyo.csv'
Tokyo_2_file = 'dataset_TSMC2013_Jan_26_Tokyo_2.csv'
#
# data2 = []
# with open(Tokyo_file, 'r', newline='', encoding='utf-8') as Tokyo_1:
#     csv_file = csv.reader(Tokyo_1)
#     for row in csv_file:
#         # print(row[0][-26:-23])  ---- Jan
#         # print(row[0][-22:-20])
#         # if row[0][-22:-20] == '17':
#         if row[0][-26:-20] == 'Jan 26':
#             data2.append(row)
#
# # 保存到新的文件中
# with open(Tokyo_1_file, 'w', newline='') as Tokyo_2:
#     Nyc_final_file = csv.writer(Tokyo_2)
#     Nyc_final_file.writerows(data2)

# list_index = [2,3,6]
# with open(Tokyo_1_file,'r',encoding='utf-8') as fr:
#     lines = fr.readlines()
#     with open(Tokyo_2_file,'w+',encoding='utf-8', newline='') as fw:
#         for line in lines:
#             temp = line.split('\t')
#             temp = [n for i, n in enumerate(temp) if i not in list_index]
#             print(temp)
#             time = temp[4]
#             time = time[4:30]
#             temp[4] = time
#             # 随机生成一个在线时间 online_time
#             online_time = random.randint(600, 2700)
#             temp[1] = online_time
#             print(temp)
#             writer = csv.writer(fw)
#             writer.writerow(temp)

"""
随机生成 10-50 个工人 和 10-50个任务
"""

# Tokyo = pd.read_csv('dataset_TSMC2013_Jan_26_Tokyo_2.csv')
#
# for tnum in range(10,60,10):
#     Tokyo_exc = Tokyo.sample(n=tnum)
#     Tokyo_exc.to_csv('Task_Jan_26_Tokyo_' + str(tnum) + '\\' + 'task_Tokyo_' + str(tnum) + '.csv', index=False, header=None)
#
# # 工人的数据
# for tnum in range(10,60,10):
#     Tokyo_exc = Tokyo.sample(n=tnum)
#     filename = 'Task_Jan_26_Tokyo_' + str(tnum) + '\\' + 'task_Tokyo_' + str(tnum) + '.csv'
#     Tokyo_1 = pd.read_csv(filename, header=None)
# # #     # del Nyc1['0']
# #     Tokyo_1.columns = ['poi_id', 'eff_time', 'lat', 'lon', 'times']
# #     print(len(Tokyo_1))
#     for j in range(len(Tokyo_1)):
#         Tokyo_1.at[j, 0]=str(j+1)
#     Tokyo_1.to_csv(filename, index=False,header=0)