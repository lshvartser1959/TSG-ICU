# TSG-ICU
Main repository of the TSG-ICU project

This code realizes the adaptive procedure and its evaluation on no-ARDS/ ARDS sets described in our paper
Title: A novel machine learning model to predict respiratory failure and invasive mechanical ventilation in critically ill patients with COVID-19
Authors: itai Aharon bendavid; Liran Statlender; Leonid Shvartser; Shmuel Teppler; Roy Azullay; Rotem Sapir; Pierre Singer
submitted to Critical Care , Supplements 3, 4

The code partly uses the MIMIC Extract framework.

TO RUN THIS CODE

1. Unzip file all_hourly_data_3000.zip 
2. Run adaptMain.py

The ADAPT function printing all the results of evaluation of the adaptiation algoritm described in the paper.

Please reference our paper in case you will use the code or the adaptation algorithm from here.

Update:

Now 3 variants of adaptaion are considered.

1) The original one
2) Oversampling with SMOTE
3) Without oversampling, but with increased portion of target data for transfer learning

The control of choice of the adaptation scheme is done with OVERSAMPL parameter in BaseFunctions.py

OVERSAMPL = 0 - no oversampling,
OVERSAMPL = 1 - RandomOverSampler,
OVERSAMPL = 2 - SMOTE.

In the current uploaded version OVERSAMPL = 0.
