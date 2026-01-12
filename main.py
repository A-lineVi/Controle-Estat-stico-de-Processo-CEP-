import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carta de Controle X (Média) e R (Amplitude).
# 5 azulejos para amostra:
n = 5
# 20 amostras coletadas:
k = 20
print("~~ Analise Quantitativa ~~")
#fatores das cartas de controle:
A2 = 0.577
D3 = 0.0
D4 = 2.114

np.random.seed(42)
data = np.random.normal(loc=0.01, scale=0.05, size=(k, n))
df = pd.DataFrame(data, columns=[f"Medição_{i+1}" for i in range(n)])

df['Media'] = df.mean(axis=1)
df['Amplitude'] = df.max(axis=1) - df.min(axis=1)

X_barra_b = df['Media'].mean()
R_barra = df['Amplitude'].mean()

LCS_X = X_barra_b + A2 * R_barra
LCI_X = X_barra_b - A2 * R_barra

LCS_R = D4 * R_barra
LCI_R = D3 * R_barra

plt.figure(figsize=(10, 5))
plt.plot(df['Amplitude'], marker='o', linestyle='-', color='orange', label='Amplitude (R)')
plt.axhline(R_barra, color='green', linestyle='-', label=f'LC (R-barra): {R_barra:.3f}')
plt.axhline(LCS_R, color='red', linestyle='--', label=f'LCS: {LCS_R:.3f}')
plt.axhline(LCI_R, color='red', linestyle='--', label=f'LCI: {LCI_R:.3f}')
plt.title('Carta de Controle R para Dispersão Dimensional') # Título corrigido
plt.xlabel('Subgrupo de Amostra (Hora)')
plt.ylabel('Amplitude (mm)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df['Media'], marker='o', linestyle='-', label='Média Amostral')
plt.axhline(X_barra_b, color='green', linestyle='-', label=f'LC (Média Geral):{X_barra_b:.3f}')
plt.axhline(LCS_R, color='red', linestyle='--', label=f'LCS: {LCS_R:.3f}')
plt.axhline(LCI_R, color='red', linestyle='--', label=f'LCI: {LCS_R:.3f}')
plt.title('Carta de controle X-Barra para desvio Dimensional')
plt.xlabel('Subgrupo de Amostra (Hora)')
plt.ylabel('Média de desvio dimensional (mm)')
plt.legend()
plt.grid(True)
plt.show()
#######################################################
print("~~ Analise Qualitativa ~~")
#total de lote inspecionado
tamanho_lote = 100 
nao_conformes = np.array([5, 3, 7, 4, 8, 5, 6, 4, 3, 7, 12, 5, 6, 4, 3, 7, 5, 6, 4, 3])
amostras = np.full(len(nao_conformes), tamanho_lote)

p_amostral = nao_conformes / amostras

p_barra = nao_conformes.sum() / amostras.sum()

sigma_p = np.sqrt(p_barra * (1 - p_barra) / tamanho_lote)
LCS_p = p_barra + 3 * sigma_p
LCI_p = max(0, p_barra - 3 * sigma_p)

plt.figure(figsize=(10, 5))
plt.plot(p_amostral, marker='o', linestyle='-', label='Proporção de Defeitos (p)')
plt.axhline(p_barra, color='green', linestyle='-', label=f'LC (p-barra): {p_barra:.3f}')
plt.axhline(LCS_p, color='red', linestyle='--', label=f'LCS: {LCS_p:.3f}')
plt.axhline(LCI_p, color='red', linestyle='--', label=f'LCI: {LCI_p:.3f}')
plt.title('Carta de Controle p para Proporção de Azulejos Não-Conformes')
plt.xlabel('Subgrupo de Amostra (Lote)')
plt.ylabel('Proporção de Azulejos Defeituosos')
plt.legend()
plt.grid(True)
plt.show()

