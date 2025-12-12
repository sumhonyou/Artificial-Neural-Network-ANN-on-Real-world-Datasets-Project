# Artificial-Neural-Network-ANN-on-Real-world-Datasets-Project
Computer Intelligence Assignment

## Overview 
- Using MLP for binary classification 
- Target: Churn 
- Output: Yes churn / No churn (1/0)
- Activation: Relu 

## Before you run the program, follow these steps
1. Ensure that you have installed jupyter notebook and python extension
2. Click the select kernel (VS code will be at your top right)
3. Choose ananconda Env or you may create a temporary virtual environment (env) [for creating the temporary virtual environment please watch YouTube video]
4. <b>Done</b> (remember to check whether need commmit or not even though you didn't do anything on the code)


## What Have We Done
- Tunning the MLP model
- Visualisation 
- Include early stopping
- Include metrics (classification, confusion matrix, training & validation linear graph analysis)


### Challenges that we have Met 
1. Dataset is unbalance (No > Yes) create bias in model inferencing
- Using Oversampling (increase number of data in the minority group) or Undersampling (reducing the number of majority class) or both 

## Metrics 
1. Confusion Matrix
2. Classification Report 
3. Accuracy Curves 
4. Loss Curves
5. ROC-AUC Curves
6. MLP Structure

## Tunning 
1. alpha rate change 0.001 to 0.005
- Have increase the recall for class 1 from 0.65 to 0.67, but for class 0 drop from 0.84 to 0.83
2. Use 1 hidden layer with 16 neuron, have best performance in terms of accuracy curves, loss function and etc

### Accuracy report 
Architecture: (16,) - 1 hidden layer with 16 neurons
Activation : relu
Train Acc  : 0.8318
Val Acc    : 0.8114

Architecture: (16, 8) - 2 hidden layers with 16 & 8 neurons 
Activation : relu
Train Acc  : 0.8482
Val Acc    : 0.7678

Architecture: (32, 16) - 2 hidden layers with 32 & 16 neurons 
Activation : relu
Train Acc  : 0.8986
Val Acc    : 0.7488

Architecture: (64, 32) - 2 hidden layers with 64 & 32 neurons
Activation : relu
Train Acc  : 0.9480
Val Acc    : 0.7545

Architecture: (32, 16, 8)
Activation : relu
Train Acc  : 0.9084
Val Acc    : 0.7517