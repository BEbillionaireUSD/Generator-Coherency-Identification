# 用于发电机同调识别

python IAE.py

**可通过调整MODEL选择不同种类的auto-encoder**

支持各种类型的时间序列聚类，包括单元和多元时序

**使用本代码请引用**：刘丰瑞,李柏橦.基于概率分布联合训练时空自编码器的同调发电机在线辨识及其可解释性[J/OL].中国电机工程学报:1-16[2021-08-13].

由于时间关系暂时没有写注释，我们会尽快把坑填上...

## Abstract
This study proposed an improved Auto-Encoder (IAE) scheme for online coherency identification by employing synchro phasor measurements obtained by phasor measurement units (PMUs) in the power system. First, deep neural network layers were constructed for feature extraction and jointly trained with the clustering layer. Second, a distance matrix that presents the similarity index between each pair of generators was calculated according to complexity-invariant distance (CID), considering the characteristic of similar rotor movement trends of coherent generator groups. Third, Gaussian Mixture Model (GMM), improved by KL divergence, was applied to implement probabilistic data-driven clustering and optimize parameters in both feature extraction and clustering processes. The proposed approach was evaluated and validated on the simulation cases of 4-machine 11-bus two-area system and 16-machine 68-bus 5-area system, as well as actual data gathered through China Southern Power Grid (CSG) and Yunnan Power Grid. Also, the high-dimensional features of power angle were visualized and explained via adopting the class activation map and the idea of transfer learning. The results demonstrate that this approach is not only robust under noise and avoid deficiency of pivotal features while mining data, but also realizes prompt and accurate online coherency identification, all the while and the explanatingon of its results’ interpretability without a manually defined number of generator groups or pre-trained off-line labeled data.

wide-area measurement; coherency identification; spatial temporal neural network; joint training; transfer learning; interpretability analysis

提出一种改进的自编码器(Improved Auto-Encoder，IAE)框架，借助电力系统广域量测信息，在线辨识电力系统同调机群。设计时空神经网络搭建特征提取层，与聚类层联合训练；考虑同调机组转子运动趋势相似的特点，计算数据特征的复杂度不变性距离(Complexity-Invariant Distance，CID)表征机组相似度；基于 Kullback-Leibler 散度(KL 散度)改进高斯混合模型(Gaussian Mixture Model，GMM)进行概率性数据驱动的聚类，优化聚类和特征提取过程中的参数。将所提方法应用于典型 4 机 11 节点 2 区域电力系统、典型16 机 68 节点 5 区域电力系统、中国南方电网进行分析、验证，基于类激活映射(Class Activation Map，CAM)和迁移学习思想进行功角高维特征的可视化解释，结果表明：该方法在噪声干扰下具有较强鲁棒性，避免了数据关键特征缺失、不依赖于人工定义的机群数量、不需要训练离线标注数据，实现了同调机群快速准确在线辨识和结果的有效解释。

## References

本代码参考了https://github.com/FlorentF9/DeepTemporalClustering 和 https://github.com/saeeeeru/dtc-tensorflow，感谢FlorentF9 和 saeeeeru
