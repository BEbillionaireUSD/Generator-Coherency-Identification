import numpy as np
from statsmodels.tsa import stattools
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from util import *

def eucl(x, y):
    """
    Euclidean distance between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    d = np.sqrt(np.sum(np.square(x - y), axis=0))
    return np.sum(d)


def cid(x, y):
    """
    Complexity-Invariant Distance (CID) between two multivariate time series given as arrays of shape (timesteps, dim)
    Reference: Batista, Wang & Keogh (2011). A Complexity-Invariant Distance Measure for Time Series. https://doi.org/10.1137/1.9781611972818.60
    """
    #assert(len(x.shape) == 2) 
    assert(x.shape == y.shape) # time series must have same length and dimensionality
    ce_x = np.sqrt(np.sum(np.square(np.diff(x, axis=0)), axis=0) + 1e-9)
    ce_y = np.sqrt(np.sum(np.square(np.diff(y, axis=0)), axis=0) + 1e-9)
    d = np.sqrt(np.sum(np.square(x - y), axis=0)) * np.divide(np.maximum(ce_x, ce_y), np.minimum(ce_x, ce_y))
    return np.sum(d)


def cor(x, y):
    """
    Correlation-based distance (COR) between two multivariate time series given as arrays of shape (timesteps, dim)
    """
    scaler = TimeSeriesScalerMeanVariance()
    x_norm = scaler.fit_transform(x)
    y_norm = scaler.fit_transform(y)
    pcc = np.mean(x_norm * y_norm)  # Pearson correlation coefficients
    d = np.sqrt(2.0 * (1.0 - pcc + 1e-9))  # correlation-based similarities
    return np.sum(d)


def acf(x, y):
    """
    Autocorrelation-based distance (ACF) between two multivariate time series given as arrays of shape (timesteps, dim)
    Computes a linearly weighted euclidean distance between the autocorrelation coefficients of the input time series.
    Reference: Galeano & Pena (2000). Multivariate Analysis in Vector Time Series.
    """
    assert (len(x.shape) == 2 and x.shape == y.shape)  # time series must have same length and dimensionality
    x_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, x)
    y_acf = np.apply_along_axis(lambda z: stattools.acf(z, nlags=z.shape[0]), 0, y)
    weights = np.linspace(1.0, 0.0, x.shape[0])
    d = np.sqrt(np.sum(np.expand_dims(weights, axis=1) * np.square(x_acf - y_acf), axis=0))
    return np.sum(d)

if __name__ == '__main__':
    '''
    four_X, four_name = load_data(file_path='PMU数据/四机.xlsx')
    four_dis = np.zeros((3, 3))
    for i in range(len(four_name)):
        for j in range(i+1, len(four_name)):
            four_dis[i][j] = cid(four_X[i], four_X[j])
            four_dis[j][i] = four_dis[i][j]
    four_dis = np.around(four_dis, 4)

    
    sixteen_12X, sixteen_name = load_data(file_path='PMU数据/16机功角.xlsx', sheet_idx=0)
    sixteen_12dis = np.zeros((15, 15))
    for i in range(len(sixteen_name)):
        for j in range(i+1, len(sixteen_name)):
            sixteen_12dis[i][j] = cid(sixteen_12X[i], sixteen_12X[j])
            sixteen_12dis[j][i] = sixteen_12dis[i][j]
    sixteen_12dis = np.around(sixteen_12dis, 4)
    
    sixteen_89X, _ = load_data(file_path='PMU数据/16机功角.xlsx', sheet_idx=1)
    sixteen_89dis = np.zeros((15, 15))
    for i in range(len(sixteen_name)):
        for j in range(i+1, len(sixteen_name)):
            sixteen_89dis[i][j] = cid(sixteen_89X[i], sixteen_89X[j])
            sixteen_89dis[j][i] = sixteen_89dis[i][j]
    sixteen_89dis = np.around(sixteen_89dis, 4)
    
    sixteen_4142X, _ = load_data(file_path='PMU数据/16机功角.xlsx', sheet_idx=2)
    sixteen_4142dis = np.zeros((15, 15))
    for i in range(len(sixteen_name)):
        for j in range(i+1, len(sixteen_name)):
            sixteen_4142dis[i][j] = cid(sixteen_4142X[i], sixteen_4142X[j])
            sixteen_4142dis[j][i] = sixteen_4142dis[i][j]
    sixteen_4142dis = np.around(sixteen_4142dis, 4)

    sixteen_4649X, _ = load_data(file_path='PMU数据/16机功角.xlsx', sheet_idx=3)
    sixteen_4649dis = np.zeros((15, 15))
    for i in range(len(sixteen_name)):
        for j in range(i+1, len(sixteen_name)):
            sixteen_4649dis[i][j] = cid(sixteen_4649X[i], sixteen_4649X[j])
            sixteen_4649dis[j][i] = sixteen_4649dis[i][j]
    sixteen_4649dis = np.around(sixteen_4649dis, 4)

    with open('CID/4机-16机.txt', 'w') as wf:
        wf.write('距离\t'+'\t'.join(['发电机'+str(_) for _ in range(2, 16)])+'\n')
        wf.write('四机\t时刻数：26041\n')
        for i in range(3):
            wf.write('发电机')
            wf.write(str(i+2)+'\t')
            wf.write('\t'.join([str(d) for d in four_dis[i]])+'\n')

        wf.write('\n16机1-2相对短路\t时刻数：1980\n')
        for i in range(15):
            wf.write('发电机')
            wf.write(str(i+2)+'\t')
            wf.write('\t'.join([str(d) for d in sixteen_12dis[i]])+'\n')
            
        wf.write('\n16机8-9相对短路\t时刻数：1980\n')
        for i in range(15):
            wf.write('发电机')
            wf.write(str(i+2)+'\t')
            wf.write('\t'.join([str(d) for d in sixteen_89dis[i]])+'\n')
        
        wf.write('\n16机41-42相对短路\t时刻数：1980\n')
        for i in range(15):
            wf.write('发电机')
            wf.write(str(i+2)+'\t')
            wf.write('\t'.join([str(d) for d in sixteen_4142dis[i]])+'\n')
        
        wf.write('\n16机46-49相对短路\t时刻数：1980\n')
        for i in range(15):
            wf.write('发电机')
            wf.write(str(i+2)+'\t')
            wf.write('\t'.join([str(d) for d in sixteen_4649dis[i]])+'\n')
    '''

    牛从, 牛从名字 = load_data(file_path='PMU数据/云南牛从22-32长负荷扰动数据.xlsx')
    niulength = 牛从.shape[0]
    牛从距离 = np.zeros((niulength, niulength))
    for i in range(niulength):
        for j in range(i+1, niulength):
            牛从距离[i][j] = cid(牛从[i], 牛从[j])
            牛从距离[j][i] = 牛从距离[i][j]
    牛从距离 = np.around(牛从距离, 4)
    牛从名字 = [s[:-2] for s in 牛从名字]

    永富, 永富名字 = load_data(file_path='PMU数据/云南永富直流双极功率升至3000MW47-53.xlsx')
    fulength = 永富.shape[0]
    永富距离 = np.zeros((fulength, fulength))
    for i in range(fulength):
        for j in range(i+1, fulength):
            永富距离[i][j] = cid(永富[i], 永富[j])
            永富距离[j][i] = 永富距离[i][j]
    永富距离 = np.around(永富距离, 4)
    永富名字 = [s[:-2] for s in 永富名字]

    永仁, 永仁名字 = load_data(file_path='PMU数据/云南永仁换流站由3000MW降为2600MW.xlsx')
    renlength = 永仁.shape[0]
    永仁距离 = np.zeros((renlength, renlength))
    for i in range(renlength):
        for j in range(i+1, renlength):
            永仁距离[i][j] = cid(永仁[i], 永仁[j])
            永仁距离[j][i] = 永仁距离[i][j]
    永仁距离 = np.around(永仁距离, 4)
    永仁名字 = [s[:-2] for s in 永仁名字]
    

    with open('CID/云南.txt', 'w') as wf:
        wf.write('牛从\t时刻数：'+str(niulength)+'\n')
        wf.write('距离\t'+'\t'.join([_ for _ in 牛从名字])+'\n')
        for i in range(niulength):
            wf.write(牛从名字[i]+'\t')
            wf.write('\t'.join([str(d) for d in 牛从距离[i]])+'\n')
        
        wf.write('永富\t时刻数：'+str(fulength)+'\n')
        wf.write('距离\t'+'\t'.join([_ for _ in 永富名字])+'\n')
        for i in range(fulength):
            wf.write(永富名字[i]+'\t')
            wf.write('\t'.join([str(d) for d in 永富距离[i]])+'\n')
        
        wf.write('永仁\t时刻数：'+str(永仁.shape[0])+'\n')
        wf.write('距离\t'+'\t'.join([_ for _ in 永仁名字])+'\n')
        for i in range(永仁.shape[0]):
            wf.write(永仁名字[i]+'\t')
            wf.write('\t'.join([str(d) for d in 永仁距离[i]])+'\n')


    南方, 南方名字 = load_data(file_path='PMU数据/南方电网实测频率.xlsx')
    length = 南方.shape[0]
    南方距离 = np.zeros((length, length))
    for i in range(length):
        for j in range(i+1, length):
            南方距离[i][j] = cid(南方[i], 南方[j])
            南方距离[j][i] = 南方距离[i][j]
    南方距离 = np.around(南方距离, 4)

    with open('CID/南方.txt', 'w') as wf:
        wf.write('南方\t时刻数：'+str(南方.shape[0])+'\n')
        wf.write('距离\t'+'\t'.join([_ for _ in 南方名字])+'\n')
        for i in range(南方.shape[0]):
            wf.write(南方名字[i]+'\t')
            wf.write('\t'.join([str(d) for d in 南方距离[i]])+'\n')
        


    
    