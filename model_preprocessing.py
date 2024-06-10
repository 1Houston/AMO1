from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def preprocess_data(data_folder):
    data = np.loadtxt(os.path.join(data_folder, 'data.csv'), delimiter=',')
    scaler = StandardScaler()
    return scaler.fit_transform(data)
if __name__ == "__main__":
    train_data_scaled = preprocess_data('train')
    test_data_scaled = preprocess_data('test')
