# -*- coding:utf-8 -*-

"""
@version: ??
@author: ‘monster‘
@site: 
@software: PyCharm
@file: Gowalla_data_exc.py
@time: 2022/10/21 18:14
"""

import time
import random
from collections import Counter

"""
Gowalla 数据集包含了2009年02月至2010年10月份之间的的用户签到信息，为了方便，我们随机选择其中某一天的签到信息（如2010-05-14）
"""
# 获取数据集中在2010-05-14的签到信息，并保存为check_in_2010_05_04.txt
def func(filename):
    writename = 'check_in_2010_05_14.txt'
    with open(filename,'r',encoding='utf-8') as fr:
        lines = fr.readlines()
        with open(writename,'w+',encoding='utf-8') as fw:
            for line in lines:
                temp = line.split('\t')
                # 用replace方法将 ’T' 和 ’Z‘ 字符分别替换成 空格 和 空字符串
                time_t = temp[1].replace('T',' ').replace('Z','')
                # 使用 time 模块的 strptime 函数将时间戳字符串转换为 time。struct_time 对象
                # 格式字符串"%Y-%m-%d %H:%M:%S"指定了时间戳字符串的期望格式
                time_geshi = time.strptime(time_t,"%Y-%m-%d %H:%M:%S")
                # 使用 time 模块的 mktime 函数将 time.struct_time 对象转换为 Unix时间戳 【浮点数】
                timestamp = time.mktime(time_geshi)
                # 检查timestamp是否落在表示2010年5月14日00:00:00 UTC到2010年5月15日00:00:00 UTC的Unix时间戳范围内
                if 1273766400 <= timestamp <= 1273852800:
                    fw.writelines(line)



def genFileName(base, t):
    base_time = base + " 00:00:00"
    time_geshi = time.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    baseTime_stamp = time.mktime(time_geshi)
    for i in range(1, 25):
        if baseTime_stamp + (i - 1) * 3600 <= t < baseTime_stamp + i * 3600:
            filename = "worker_05_14_by_hour\\checkin_2010_05_14_0" + str(
                i) + "_arrive_and_leave.txt" if i < 10 else "worker_05_14_by_hour\\checkin_2010_05_14_" + str(i) + "_arrive_and_leave.txt"
            break
    return filename

# 随机的生成每个小时内的参与者到达和参与者离开时间：
# 文件名：checkin_05_14_{01}_arrive_and_leave.txt (注 {01} 表示00：00 - 01：00)
# 文件格式： user_id  arrive_time leave_time latitude longitude location_id
# 主要目的是将一个原始数据文件分割成多个按小时为单位的文件，并为每个工人生成随机的在线时间。
def genWorkerByHour(filename):
    with open(filename,'r',encoding='utf-8') as rf:
        lines = rf.readlines()
        for line in lines:
            """# 工人在线时间 10min-45min"""
            online_time = random.randint(600, 2700)
            temp = line.split('\t')
            time_t = temp[1].replace('T', ' ').replace('Z', '')
            time_geshi = time.strptime(time_t, "%Y-%m-%d %H:%M:%S")
            timestamp = time.mktime(time_geshi)
            writeFileName = genFileName('2010-05-14',timestamp)
            with open(writeFileName,'a+',encoding='utf-8') as wf:
                text = temp[0] + '\t' + str(timestamp) + '\t' + str(timestamp + online_time) + '\t' + temp[2] + '\t' + \
                       temp[3] + '\t' + temp[4]
                wf.writelines(text)
                wf.close()


# 从每个小时的签到数据中，随机选择参与者，参与者人数范围为[10,100]，步长10
# 从每个小时的签到数据中，选择最流行的位置作为任务的位置， 任务数量的范畴[10,100],步长10
def selectWorkerAndTask(filename,wnums,wstep,tnums,tstep):
    """

    :param filename: 文件名称
    :param nums: 参与者人数范围  [10,90]
    :param step: 步长  10
    :return:
    """
    with open(filename,'r',encoding='utf-8') as rf:
        filename = filename.split('/')[1]
        # 预选去除重复的ID
        lines = rf.readlines()
        # 去除重复的工人id
        workid = set()
        # 记录每个地点被签到的次数
        placeid = Counter()
        # temp_check_in_record 列表中保存了所有没有重复工人 ID 的数据行。
        temp_check_in_record = []
        for line in lines:
            # strip() 去掉行末的空白符
            item = line.strip().split('\t')
            # item[-1] 列表最后一个元素作为 键，是位置 id：统计每个地点出现的次数
            placeid[item[-1]] += 1
            if item[0] in workid:
                continue
            workid.add(item[0])
            temp_check_in_record.append(line)

        # 生成任务概貌：将生成的任务信息写入文件中
        # 统计所有签到记录中出现的地点以及他们出现的次数
        tlen = len(placeid)
        for tnum in range(tnums[0],tnums[1],tstep):
            if tnum>tlen:
                print(f"数据不足，采用所有签到数据作为任务信息")
                # temp_place:是一个列表，存储出现频率最高的 tnum 个地点 及 出现次数
                # 使用 most_common() 方法从 Counter 对象中获取
                temp_place = placeid.most_common(tlen)
            else:
                temp_place = placeid.most_common(tnum)
            # temp_place_id 是一个列表，存储 temp_place 中的地点 ID
            temp_place_id = [temp_place[idd][0] for idd in range(len(temp_place)) ]
            writename_t = 'task_05_14_by_hour_'+str(tnum)+'\\'+filename[0:22]+'task.txt'
            # a+是以追加模式打开文件，如果文件不存在则创建文件。在追加模式下，文件指针位于文件末尾，
            # 新的内容将被写入到已有内容的后面。如果文件已经存在，则新的内容将追加到文件末尾。
            with open(writename_t, 'a+' ,encoding='utf-8') as twf:
                # 创建一个空的集合 ： 用于记录已经添加过的任务
                have_add_task = set()
                for line in lines:
                    # 将一条签到记录拆分为列表ll
                    ll = line.strip().split('\t')
                    # 判断当前签到记录对应的任务ID是否在筛选出来的任务ID列表temp_place_id中，
                    # 并且该任务ID不在已添加的任务集合have_add_task中
                    if ll[-1] in temp_place_id and ll[-1] not in have_add_task:
                        # 删除了签到时间
                        # del ll[2]
                        writeText = '\t'.join(ll)+'\n'
                        have_add_task.add(ll[-1])
                        twf.writelines(writeText)
        # 生成工人概貌
        length = len(workid)
        for num in range(wnums[0],wnums[1]+1,wstep):
            if length< num:
                print(f"工人数量不足{num},采用全部的工人")
                temp_id = random.sample(range(0,length),length)
            else:
                temp_id = random.sample(range(0,length),num)
            writename_w = 'worker_05_14_by_hour_'+str(num)+'\\'+filename
            with open(writename_w,'a+',encoding='utf-8') as wwf:
                for t_id in temp_id:
                    wwf.writelines(temp_check_in_record[t_id])

def genWorkAndTask():
    for i in range(1,25):
        if i<10:
            filename1 = 'worker_05_14_by_hour/checkin_2010_05_14_0'+str(i)+'_arrive_and_leave.txt'
        else:
            filename1 = 'worker_05_14_by_hour/checkin_2010_05_14_'+str(i)+'_arrive_and_leave.txt'
        selectWorkerAndTask(filename1,[10,100],10,[10,100],10)

class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    # func('Gowalla_totalCheckins.txt')
    # genWorkerByHour('check_in_2010_05_14.txt')
    genWorkAndTask()