# ProbCIFI: Generator Coherency Identification

Support python 3.7
pip install -r requirements.txt
python IAE.py

**Support PMU time series clustering with different kinds of auto-encoders**


## Abstract

We propose a novel end-to-end generator coherency identification framework, which integrates an improved Auto-Encoder to comprehensively exploit information obtained by phasor measurement units. It employs the multi-task learning technique by jointly training the feature extraction module and the clustering layer, fully exploring the relationship between the two tasks and obtain cluster-specific temporal representations. The clustering phase is conversed into a supervised probabilistic multi-class categorization problem and implemented by a Gaussian Mixture Model variant. In addition, the attended features of power angles are visualized through the class activation map and the idea of transfer learning, empowering natural interpretability to the agnostic learning process. Simulation studies demonstrate the effectiveness and validation of the proposed generator coherency identification scheme, which is insensitive to the window size or noise, outperforming existing state-of-the-art methods across datasets from different situations.


## References

https://github.com/FlorentF9/DeepTemporalClustering 和 https://github.com/saeeeeru/dtc-tensorflow
Tnank FlorentF9 and saeeeeru
