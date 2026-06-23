import numpy as np
import pandas as pd

# Carrega a base de dados a partir de seu caminho
data = np.load('data.npz')
X_test = data['X_test']
X_train = data['X_train']
Y_train = data['Y_train']

# oRF
num_samples = X_test.shape[0]
#y_hat = 

# Gera um DataFrame no formato esperado da submissão
submission_df = pd.DataFrame({
    'ID': np.arange(1, num_samples + 1),
    'Prediction': y_hat
})

# Salva o arquivo CSV no diretório atual
submission_df.to_csv('submission.csv', index=False)
