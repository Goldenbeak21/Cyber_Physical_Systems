# Multi-Variate Time Series Analysis using LSTM/GRU

## Summary
- Developed a dual attention based LSTM and GRU models to pre-emtively detect anomalies in a power plant. 
- Used spatial input and temporal attention to extract important features and thereby reduced the false alarm rate and improved the detection accuracy. 
- Analyse the feature importance by investigating the weights assigned by the attention layers. 
- Compare the performance of the deep learning based techniques with machine learning algorithms. 

## Files
- **Dataset Creation:** Attack_Dataset_Creation.py, reshaping.py - Used for pre-processing the raw data to create a dataset for training the ML/DL models. 
- **Deep Learning Models:** LSTM.py, DA_LSTM.py (Dual Attention Based LSTM), time_series_attack_detection.py - Deep Learning models for detecting anomalies in time series data. 
- **Machine Learning Models:** CUSUM.py, SVR.py - Implement traditional statistical anomaly detection techniques and machine learning techniques. 
- **Performance Metrics:** custom_metrics.py, plotting_.py - Analysing and comparing the performance of various models. 

**NOTE:** This research work was published at IEEE SmartGridComm. [S. Ghahremani, R. Sidhu, D. K. Y. Yau, N. -M. Cheung and J. Albrethsen, "Defense against Power System Time Delay Attacks via Attention-based Multivariate Deep Learning," 2021 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm), Aachen, Germany, 2021, pp. 115-120, doi: 10.1109/SmartGridComm51999.2021.9632305.](https://ieeexplore.ieee.org/abstract/document/9632305) 
