import numpy as np
import pandas as pd
from tqdm import tqdm

steps = 10000
learning_rate = 0.0002

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def predict(X, W):
    z = np.dot(X, W)
    return sigmoid(z)

def parameter_estimation(dataframe):
    num_features = dataframe.shape[1] - 1
    
    # Convert dataframe to numpy array 
    data = dataframe.values
    
    # Initialize thetas with an extra element for bias
    thetas = np.random.rand(num_features + 1)
    
    for _ in tqdm(range(steps)):
        gradients = np.zeros(num_features + 1)
        
        for row in data:
            X = row[:-1]  # Features
            y = row[-1]   # Label
            
            # Add bias term to features
            X = np.insert(X, 0, 1)
            
            # predict

            prediction = predict(X, thetas)

            # Update gradients
            for index,values in enumerate(X):
                gradients[index] += (y-prediction)*values
                
        # Update thetas
        thetas += learning_rate * gradients
    
    return thetas

def classify(X, W, threshold):
    # Add bias term to features
    X = np.insert(X, 0, 1)
    
    if predict(X, W) > threshold:
        return 1
    else:
        return 0

def main():
    training_path = 'Codes/Python/Stanford Psets/Pset6/data/netflix-train.csv' # Put your training .csv file path here
    training_df = pd.read_csv(training_path)
    training_df.drop("Demographic",axis=1,inplace=True)
        
    params = parameter_estimation(training_df)
    testing_path = 'Codes/Python/Stanford Psets/Pset6/data/netflix-test.csv'  # Put your testing .csv file path here
    testing_dataframe = pd.read_csv(testing_path)
    testing_dataframe.drop('Demographic',axis=1,inplace=True)
    print(params)

    accuracy = 0
    for index, row in testing_dataframe.iterrows():
        X_test = row[:-1].values  # Features (excluding label)
        y_test = row['Label']  # Label
        
        if classify(X_test, params, 0.5) == y_test:
            accuracy += 1
    
    print('accuracy:', accuracy / len(testing_dataframe))

if __name__ == '__main__':
    main()



"""
Params for Heart Train.csv: 
1st is bias.
[-0.04267873  0.09439218  0.80325869  0.33120237  0.08047857  0.87299393
  0.87473926  0.51373267  0.473024    0.61935925  0.3258522   0.46677625
  0.57871476  0.27963092  0.17400409  0.04461191  0.01018244  0.42791911
  0.70330562  0.3379361   0.20079681 -0.0156292   0.6022028   0.26817905]
"""

"""
Params for simpletrain.csv

[0.16562933 0.66736026 0.55875267]

"""
"""
Params for netflix.csv

[-1.53026782e+00  2.27224786e-01 -6.46238237e-04 -1.32040296e-01
 -9.91159047e-02  2.52158756e-01  2.86539638e-03 -2.56549917e-02
  2.07797876e-01 -1.27764689e-02 -9.23999152e-02  8.29330515e-02
  4.40164550e-02  1.69074113e-01 -7.79385320e-02 -2.88810124e-02
 -2.79895281e-02 -1.34292568e-02  2.21425727e-01  1.86446175e+00
  5.83075325e-02]
"""

"""
Params for ancestry-train.csv:

[ 1.13084407 -1.3895138   1.03122535 -0.88320413  2.30536982  0.73559915
  2.51560737  0.99957248 -1.44463103  1.68925447 -0.22956707  2.32787167
 -1.73658146  1.50850205  2.13025599  1.15238018 -2.04389914 -0.72833463
 -4.10437476  0.70349575 -1.9118951 ]

"""
