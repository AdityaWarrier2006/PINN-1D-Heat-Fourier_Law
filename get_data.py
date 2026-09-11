import os
import scipy.io
import numpy as np
import pandas as pd
import urllib.request

def get_data():
    file_path = '1D_Heat_Synthetic_Data.mat'
    
    if not os.path.isfile(file_path):
        print("Did not find", file_path)
        url = "https://github.com/AdityaWarrier2006/PINN-1D-Heat-Fourier_Law/raw/refs/heads/main/1D_Heat_Synthetic_Data.mat"
        urllib.request.urlretrieve(url, file_path)
        print("Downloaded file.")
    else:
        print("File already exists...")

    mat_data = scipy.io.loadmat(file_path)
    x = mat_data['x'].flatten()
    t = mat_data['t'].flatten()
    T = mat_data['T']

    X, Time = np.meshgrid(x, t, indexing='ij')
    df = pd.DataFrame({
        'x': X.flatten(),
        't': Time.flatten(),
        'T': T.flatten()
    })

    return df.sample(n=1000, random_state=42)

if __name__ == "__main__":
    df_sampled = get_data()
    print(df_sampled.head())
    print(df_sampled.shape)
