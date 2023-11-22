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
from collections import deque, defaultdict
import time
# 记录开始时间
start_time = time.time()
INF = 999999
# 速度暂时设定 4m/s
v = 4
scale = 1

def rad_of_deg(ang):
    return ang * math.pi / 180  # ?
def deg_of_rad(ang):
    return ang * 180 / math.pi

RADIANT_TO_KM_CONSTANT = 6371.0088
LATITUDE_TO_M_CONSTANT = 111000
LONGITUDE_AT_EQUATOR_IN_M_CONSTANT = 111321

def get_distance_in_meters(lat1, lon1, lat2, lon2):
    lat_dist_in_m = (lat2-lat1)*LATITUDE_TO_M_CONSTANT
    lon_dist_at_lat = LONGITUDE_AT_EQUATOR_IN_M_CONSTANT * (math.cos(rad_of_deg(lat1)) + math.cos(rad_of_deg(lat2))) / 2 # using average. should be ok since only looking at short distances
    lon_dist_in_m = (lon2-lon1) * lon_dist_at_lat
    distance = round(np.sqrt(lon_dist_in_m**2 + lat_dist_in_m**2)) # using pythagoras
    return distance/scale

#####################################
# 需要获取每个工人的可完成任务子集
#####################################

# 获取工人的数据
w_data = pd.read_csv('Worker_Jan_26_Tokyo_20\worker_Tokyo_20_0.7.csv', header=0)
w_data = np.array(w_data)
# print(w_data[0][1])

w_num = len(w_data)
print(w_num)

# 获取任务的数据
t_data = pd.read_csv('Task_Jan_26_Tokyo_10\\task_Tokyo_10.csv', header=None)
t_data = np.array(t_data)
# print(t_data)

t_num = len(t_data)
print(t_num)

# 创建图
G = nx.Graph()
# 工人到任务
for i in range(w_num):
    for j in range(t_num):
        # dis = round(math.sqrt((w_data[i][2]-t_data[j][2])**2 + (w_data[i][3]-t_data[j][3])**2), 5)
        dis = get_distance_in_meters(t_data[j][2],t_data[j][3],w_data[i][2],w_data[i][3])
        # print(t_data[j][2],t_data[j][3],w_data[i][2],w_data[i][3])
        # print(dis)
        # # 加入符合要求的边:工人到第一个任务节点间可到达
        # if (dis * 1000 <= w_data[i][1]  * v) and (dis * 1000 <= t_data[j][1]  * v):
        if (dis <= w_data[i][1] * v) and (dis <= t_data[j][1] * v):
            G.add_edge(i, j + w_num, weight=dis)

# 任务到任务
for i in range(t_num):
    for j in range(i+1, t_num):
        # round() 返回浮点数的四舍五入值
        # dis = round(math.sqrt((t_data[i][2]-t_data[j][2])**2 + (t_data[i][3]-t_data[j][3])**2), 5)
        dis = get_distance_in_meters(t_data[j][2], t_data[j][3], t_data[i][2], t_data[i][3])
        # 加入符合要求的边:任务节点间可到达
        # if (dis * 1000 <= t_data[i][1] * v) and (dis * 1000 <= t_data[j][1] * v):
        if (dis <= t_data[i][1] * v) and (dis <= t_data[j][1] * v):
            G.add_edge(i + w_num, j + w_num, weight=dis)


"""
构建一个有效时间表
"""


"""
# 使用邻接矩阵来存储各个位置的距离关系
"""
# G_num = G.number_of_nodes()
# print(G_num)

global NewGraph
# 初始化 邻接矩阵
NewGraph = np.full((w_num + t_num, w_num + t_num),INF,dtype=float)

# print(NewGraph)
# # 将图的点、边、权重信息加入到邻接矩阵中
for edge in G.edges():
    m, n = edge
    # print(G.get_edge_data(m,n)['weight'])
    NewGraph[m][n] = G.get_edge_data(m,n)['weight']
    # print(NewGraph[m][n])
    NewGraph[n][m] = G.get_edge_data(m,n)['weight']
# print(NewGraph)


"""
# 改进版深度遍历 查找所有在 约束条件下 可到达的路径
"""
def dfs_task_set(adj_matrix, start, end, worker_val_time, visited, path, all_paths, path_length):
    # 标记当前节点已被访问
    visited[start] = True
    # 把当前节点加入路径
    path.append(start)

# 如果当前节点是终点，则将路径加入到所有路径中
    if start == end:
        # 如果 整条路径 超出 工人的有效活动时间 就不会加入路径列表
        # if len(path) != 1 and (path_length * 1000) <= worker_val_time * v:
        if len(path) != 1 and path_length <= worker_val_time * v:
            if len(all_paths) == 0:
                # print(path)
                key = tuple(sorted(path))
                all_paths[key]= [path.copy()[1:], len(path.copy()[1:]), path_length]
                paths[key] = [path.copy(), len(path.copy()[1:]), path_length]
            else:
                key = tuple(sorted(path))
                if key not in all_paths.keys():
                    all_paths[key] = [path.copy()[1:], len(path.copy()[1:]), path_length]
                    paths[key] = [path.copy(), len(path.copy()[1:]), path_length]
                else:
                    if all_paths.get(key)[1] > path_length:
                       all_paths[key] = [path.copy()[1:], len(path.copy()[1:]), path_length]
                       paths[key] = [path.copy(), len(path.copy()[1:]), path_length]
            # 可以加入路径 + 路径 dis
            # 去重 操作
    else:
        #  遍历当前节点的所有邻居节点
        for i in range(len(adj_matrix[start])):
            # print(len(adj_matrix[start]))
            # 如果邻居节点没有被访问过且有边，则继续遍历
            if adj_matrix[start][i] != INF and not visited[i] and i >= w_num:
                # 保证整条路经每个任务都在其有效时间内完成
                # w_num 工人数量
                # 任务时间
                # if (path_length + adj_matrix[start][i]) * 1000 <= t_data[i-w_num][1] * v:
                if (path_length + adj_matrix[start][i]) <= t_data[i - w_num][1] * v:
                    path_length += adj_matrix[start][i]
                    dfs_task_set(adj_matrix, i, end, worker_val_time, visited, path, all_paths, path_length)
                    path_length -= adj_matrix[start][i]
    # 将当前节点从路径中删除，回溯到上一个节点
    path.pop()
    visited[start] = False

# 获取每条边的权重等属性 返回一个三元组的列表
# print(G.edges(data=True))

# 数据设置
visited = [False] * len(NewGraph)

path = []
all_paths = {}
paths = {}
path_length = 0

# 求取所有路径
for i in range(w_num):
    worker_val_time = w_data[i][1]
    # print(worker_val_time)
    for j in range(w_num, w_num + t_num):
        dfs_task_set(NewGraph, i, j, worker_val_time, visited, path, all_paths, path_length)

print(all_paths)
print(paths)
"""
# 对字典路径进行处理 提取键值出来存储到一个列表中
"""
valid_path = []
for i in paths.keys():
    valid_path.append(paths[i])
print(valid_path)

# valid_path = []
# for i in all_paths.keys():
#     valid_path.append(all_paths[i])
# print(valid_path)




# 字典子集
TaskSet = {}
# 数组子集 :去除掉了首字母？
taskset = []

# 获取任务子集
for per_path in valid_path :
    key = tuple(sorted(per_path[0][1:]))
    # print(sorted(per_path[0][1:]))
    if key not in TaskSet.keys():
        TaskSet[key] = per_path[1]
        taskset.append([sorted(per_path[0][1:]), per_path[1]])
# print(taskset)
taskset_num = len(TaskSet)
# 任务子集的个数
# print(taskset_num)


"""
# MCMF 算法思想板块
"""

"""
# 冲突检测1: 任务集中任务不能重复 因为任务是一次性交付任务
"""
def taskset_conflit1(cur_taskset, cur_node):
    # 有冲突
    if set(cur_taskset) & set(taskset[cur_node-w_num-1][0]):
        return 0
    else:
        # 无冲突
        return 1

"""
# 冲突检测2： 工人只要选择了一个任务，就不能再多选其分支下的另外的任务
# 冲突检测2：工人选过之后不能再重复了
"""
def workerset_conflit2(cur_workerset, cur_worker):
    # 有冲突
    if cur_worker in cur_workerset:
        return 0
    else:
        # 无冲突
        return 1


"""
函数返回值：
prev_v：列表，prev_v[v] 表示在最短路径 节点 v 的前一个节点；
prev_e：列表，prev_e[v] 表示最短路径节点 v 与 prev_v[v] 之间的边的编号；
dist：列表，dist[v] 表示从源点 s 到节点 v 的最短距离。
"""

def spfa(graph, s, t):
    n = len(graph)
    # print("------------------")
    # print(n)
    # print("------------------")
    dist = [float('inf')] * n
    dist[s] = 0
    prev_v = [-1] * n
    prev_e = [-1] * n
    in_queue = [False] * n
    in_queue[s] = True
    q = deque([s])
    cur_taskset = []
    cur_workerset = []
    while q:
        u = q.popleft()
        in_queue[u] = False
        # 遍历 u 的出边
        if u >= 1 and u <= w_num:
            if workerset_conflit2(cur_workerset, u) == 0:
                break
            else:
                cur_workerset.append(u)
        for v, cap, cost, _ in graph[u]:
            if cap > 0 and dist[v] > dist[u] + cost:
                if v >= w_num + 1 and v <= n - 2:
                    if cur_taskset == []:
                        for i in range(len(taskset[v-w_num-1][0])):
                            cur_taskset.append(taskset[v-w_num-1][0][i])
                        dist[v] = dist[u] + cost
                        prev_v[v] = u
                        prev_e[v] = _
                        if not in_queue[v]:
                            q.append(v)
                            in_queue[v] = True
                    else:
                        if taskset_conflit1(cur_taskset, v) == 0:
                            continue
                        else:
                            for i in range(len(taskset[v-w_num-1][0])):
                                cur_taskset.append(taskset[v-w_num-1][0][i])
                            dist[v] = dist[u] + cost
                            prev_v[v] = u
                            prev_e[v] = _
                            if not in_queue[v]:
                                q.append(v)
                                in_queue[v] = True
                else:
                    dist[v] = dist[u] + cost
                    prev_v[v] = u
                    prev_e[v] = _
                    if not in_queue[v]:
                        q.append(v)
                        in_queue[v] = True

    return prev_v, prev_e, dist

# cur_taskset1=[]

def dfs_mcmf(graph, s, t, f, prev_v, dist):
    global minCost
    minCost = 0
    # cur_workerset1 = []
    if s == t:
        return f
    # 倒回去添加 cost
    for i, (v, cap, cost, _) in enumerate(graph[s]):
        if cap > 0 and prev_v[v] == s and dist[v] == dist[s] + cost:
            df = dfs_mcmf(graph, v, t, min(f, cap), prev_v, dist)
            if df > 0:
                graph[s][i][1] -= df
                graph[v][graph[s][i][3]][1] += df
                minCost += cost
                if graph[s][i][0] >= 1 and graph[s][i][0] <= w_num:
                    graph[s][i][1] = 0
                # print(minCost)
                return df
    return 0

def mcmf_dinic_spfa(graph, s, t):
    # n = len(graph)
    flow = 0
    cost = 0
    while True:
        prev_v, prev_e, dist = spfa(graph, s, t)
        # 没有增广路时，跳出循环
        if prev_v[t] == -1:
            break
        # 计算增广流量和费用
        f = dfs_mcmf(graph, s, t, float('inf'), prev_v, dist)
        if f > 0:
        # 更新流量和费用
            flow += f
            # print(flow)
            cost += minCost
            # print(cost)
        else:
            break
        cost = round(cost, 5)
    return flow, cost

# 邻接表存储网络流图
def add_edge_from(graph, u, v, cap, cost):
    # 添加正向边
    graph[u].append([v, cap, cost, len(graph[v])])
    # 添加反向边
    graph[v].append([u, 0, -cost, len(graph[u])-1])

# 用于判断当前脚本是否在直接执行，或者是被其他脚本引入后执行。
if __name__ == '__main__':
    Graph_mcmf = defaultdict(list)

    """
       # 构建网络流图 节点计数
       S:0
       worker: 1 --- w_num
       task set: index + w_num + 1 --- taskset_num + w_num
       T: w_num + 1 + taskset_num
    """
    # 添加源点到工人集的边 + 容量 + cost
    for worker_index in range(1,w_num+1):
        add_edge_from(Graph_mcmf, 0, worker_index, t_num, 0)

    # 添加工人到任务子集之间的边 + 容量 + cost
    for index1 in range(len(valid_path)):
        for index2 in range(len(taskset)):
            if sorted(valid_path[index1][0][1:]) == taskset[index2][0]:
                # print(valid_path[index][0][0])
                add_edge_from(Graph_mcmf, valid_path[index1][0][0], index2 + w_num + 1, valid_path[index1][1], valid_path[index1][2])

    # 添加任务集到汇点的边 + 容量 + cost
    for task_index in range(len(taskset)):
        # 任务集节点 字典存储 任务子集元素
        # task_nodes[task_index+4] = taskset[task_index][0];
        add_edge_from(Graph_mcmf, task_index + w_num + 1, w_num + taskset_num + 1, taskset[task_index][1], 0)

    # print(w_num + taskset_num + 1)
    maxflow, mincost = mcmf_dinic_spfa(Graph_mcmf, 0, w_num + taskset_num + 1)
    print(maxflow)
    print(mincost)

# 绘制图形
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
# 记录结束时间
end_time = time.time()

# 计算实验的执行时间
execution_time = end_time - start_time

print(f"实验执行时间: {execution_time} 秒")