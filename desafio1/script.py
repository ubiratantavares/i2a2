from numpy import where
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 30)

# cria um dataframe a partir da leitura do arquivo excel
df = pd.read_excel('defective_equipment.xlsx', index_col=0, sheet_name='Data')

# cria um dataframe transposto (colunas do dataframe viram linhas e as linhas viram colunas)
df_transposto = df.T

# cria o objeto para a transformação considerando 2 componentes
pca = PCA(n_components=2)

# prepara a transformação no dataframe transposto
pca.fit(df_transposto)

# aplica a transformação do pca no dataframe transposto
df_transposto_pca = pca.transform(df_transposto)

# cria objeto para identificar outlier no dataframe transposto transformado com o pca
iso = IsolationForest(contamination=0.1)

predictions = iso.fit_predict(df_transposto_pca)

outlier_index = where(predictions == -1)

df_transform_pca_outlier = df_transposto_pca[outlier_index]

labels = list(df.columns)

plt.scatter(df_transposto_pca[:, 0], df_transposto_pca[:, 1], color='blue', label='Equipamento Não-Defeituoso')
plt.scatter(df_transform_pca_outlier[:, 0], df_transform_pca_outlier[:, 1], color='red', label="Equipamento Defeituoso")
for i, label in enumerate(labels):
    plt.annotate(label, (df_transposto_pca[i, 0], df_transposto_pca[i, 1]))
plt.grid()
plt.legend()
plt.savefig('analise.png', format='png')
plt.show()

