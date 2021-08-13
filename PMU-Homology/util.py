#! -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

def add_noise(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    #plt.plot(noise)
    #plt.show()
    return np.array(x)+np.array(noise)


def load_data(file_path='PMU数据/四机.xlsx', sheet_idx=0, time=[], time_interval=0, snr=0, show=False):
    '''
    time: 数据的起始和终止点
    time_interval: 每隔若干时刻取一个步长
    snr: 信噪比
    '''
    names = []
    print("******** 正在处理文件 "+file_path[6:-5]+"****** ")
    if '.csv' in file_path:
        df = pd.DataFrame(pd.read_csv(file_path))
    elif '.xlsx' in file_path:
        df = pd.DataFrame(pd.read_excel(file_path, sheet_name=sheet_idx))
    for key in df.keys():
        if '时间' in str(key) or '参考机' in str(key):
            continue
        names.append(key)

    samples = len(names)
    data = []

    for key in names:
        if snr>0:
            data.append(add_noise(df[key][:], snr))
        else:
            data.append(df[key][:])

    data = np.array(data).astype('float64')

    if len(time) > 0:
        data = data[:, time[0]:time[1]]
    times = data.shape[1]

    if time_interval > 0:
        assert(times % time_interval == 0)
        ds = []
        for sample in data:
            ds.append([np.mean(sample[i:i+time_interval])
                       for i in range(0, times, time_interval)])
        data = np.array(ds)
        times = int(times/time_interval)

    print('样本数量:', samples, '时刻数:', times)
    if show:
        return data, names
    dataSet = data.reshape(samples, times, 1).astype('float64')
    X = TimeSeriesScalerMinMax().fit_transform(dataSet)
    return X, names

