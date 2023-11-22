import sys
import pandas as pd
from numpy.random import random_sample
from scipy.special import lambertw
import math
import numpy as np
import csv
import networkx as nx
import matplotlib.pyplot as plt
import os


"""
Lambertw 函数：乘积对数函数
"""

assert (sys.version_info[0]==3), "This is a Python3.X code and you probably are using Python2.X"

#########################################
#
# Add the generated noise directly on the
# gps coordinates
# Don't modify this!
# 将原始纬度和经度从度数转换为弧度数，便于后续计算。
# 根据距离和方位角，使用三角函数计算出新位置的纬度（lat2）和经度（lon2）。
# 将计算出的新经度值归一化到 -180 到 180 的范围内（因为经度的范围是 -180 到 180）。
# 将新的纬度和经度从弧度转换为度数，作为最终结果返回。
#########################################
def addVectorToPos(original_lat, original_lon, distance, angle):
    # 将距离转换为弧度单位 ang_distance:表示转换后的距离
    ang_distance = distance / RADIANT_TO_KM_CONSTANT
    # lat1 和 lon1分别表示转换后的经度和纬度
    lat1 = rad_of_deg(original_lat)
    lon1 = rad_of_deg(original_lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(ang_distance) +
                     math.cos(lat1) * math.sin(ang_distance) * math.cos(angle))
    lon2 = lon1 + math.atan2(
        math.sin(angle) * math.sin(ang_distance) * math.cos(lat1),
        math.cos(ang_distance) - math.sin(lat1) * math.sin(lat2))
    lon2 = (lon2 + 3 * math.pi) % (2 * math.pi) - math.pi  # normalise to -180..+180
    # 最终返回终点的经纬度坐标
    return deg_of_rad(lat2), deg_of_rad(lon2)

#############################################
# Useful for the addVectorToPos function
# Don't modify this!
############################################
# 角度 转为 弧度
def rad_of_deg(ang):
    return ang * math.pi / 180
# 弧度 转为 角度
def deg_of_rad(ang):
    return ang * 180 / math.pi

# param:噪声的参数
# 输出的是一个包含两个值的元组:极径\极角
def compute_noise(param):
    # 隐私预算
    epsilon = param
    # theta:[0,2Π]之间的随机角度.
    # 极坐标系中的随机角度:random_sample():生成一个0到1之间的随机数
    theta = random_sample() * 2 * math.pi
    # 求取极径
    r = -1. / epsilon * (np.real(lambertw((random_sample() - 1) / math.e, k=-1)) + 1) #np.real返回复杂参数的实部
    return r, theta

#############################################
# Constants used in the implementation
# Don't modify this!
############################################

epsilon = 1.7


#########################################################
#                                                       #
#  Task 3: Apply Location Privacy Protection Mechanism  #
#                                                       #
##########################################################
# apply Geo-Indistinguishability
# noisy_latitude= []
# noisy_longitude= []
def obufused_worker(num, hour, worker_data, lat, lon, epsilon):
    # obu_worker = 'worker_moni_' + str(num) + '_' + str(epsilon) + '.csv'
    obu_worker = 'worker_05_14_by_hour_' + str(num) + '\\' +'checkin_2010_05_14_0'+ str(
                hour) + "_arrive_and_leave_" +str(epsilon)+ ".csv" if hour < 10 else "worker_05_14_by_hour_" + str(num) + '\\' +"checkin_2010_05_14_" + str(hour) + "_arrive_and_leave_" +str(epsilon)+ ".csv"
    with open(obu_worker, 'a+', newline='') as obuworker_csvfile:
        writer = csv.writer(obuworker_csvfile)
        writer.writerow(["poi_id", "eff_time", "lat", "lon","times"])
        for user in range(len(worker_data)):
            r, theta = compute_noise(epsilon)
            print(lat[user])
            print('....')
            print(lon[user])
            lat_noise, lon_noise = addVectorToPos(lat[user], lon[user], r, theta)
            # write output (with same precision as in original data)
            noisy_lat = round(lat_noise, 5)   #方法将返回lat_noise的值，该值四舍五入到小数点后的5位数字
            print(noisy_lat)
            noisy_lon = round(lon_noise, 5)
            print(noisy_lon)
            print({noisy_lat,noisy_lon})
            writer.writerow([worker_data[user][0], worker_data[user][1], noisy_lat, noisy_lon, worker_data[user][4]])

    # noisy_latitude.append(noisy_lat)
    # noisy_longitude.append(noisy_lon)

######################################
# 工人的数据 taxi_data
# 先用模拟数据 ----> 再用真实数据集
# 读取 CSV文件, 将其中"poi_id" "lat" "lon"三列转换为 数字类型
# 应该还有个 有效时间 列
######################################
for num in range(10, 60, 10):
    for hour in range(1, 25, 1):
        print(num)
        print(hour)
        # real_worker = 'worker_moni_' + str(num) + '.csv'
        real_worker = 'worker_05_14_by_hour_' + str(num) + '\\' +'checkin_2010_05_14_0'+ str(
                hour) + "_arrive_and_leave.csv" if hour < 10 else "worker_05_14_by_hour_" + str(num) + '\\'+"checkin_2010_05_14_" + str(hour) + "_arrive_and_leave.csv"
        df1 = pd.read_csv(real_worker, header=0)
        # pd.to_numeric 函数将列转换为数字类型
        df1["poi_id"] = pd.to_numeric(df1["poi_id"])
        df1["lat"] = pd.to_numeric(df1["lat"])
        df1["lon"] = pd.to_numeric(df1["lon"])
        lat = df1["lat"].values
        lon = df1["lon"].values
        taxi_data = np.array(df1)  # array 创建一个数组
        print(taxi_data)
        # errors='coerce' : 表示如果某个单元格无法转换为数字,则将其转换为 NaN
        df1 = df1.apply(pd.to_numeric, errors='coerce')
        obufused_worker(num, hour, taxi_data, lat, lon, epsilon)

#    poi_id  lon  lat
# 0       1    0    0
# 1       2    1    5
# 2       3    2    3
# print(df1)

###load POI location data###
# df2 = pd.read_csv('task_moni_10.csv', header=0)
# df2["poi_id"] = pd.to_numeric(df2["poi_id"])
# df2["lat"] = pd.to_numeric(df2["lat"])
# df2["lon"] = pd.to_numeric(df2["lon"])
# poi_data = np.array(df2) #array 创建一个数组
# print(poi_data)
# print("data loaded")







