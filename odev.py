import pandas as pd

# Excel dosyasını yükleme
file_path = 'Ogrenci_Performans.xlsx'
df = pd.read_excel(file_path)

# Verinin ilk birkaç satırını görüntüleme
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Temel istatistikler
print(df.describe())

# Cinsiyet ve başarı arasındaki ilişki
sns.boxplot(x='Cinsiyet', y='Matematik', data=df)
plt.title('Cinsiyete Göre Matematik Başarısı')
plt.show()

sns.boxplot(x='Cinsiyet', y='Okuma', data=df)
plt.title('Cinsiyete Göre Okuma Başarısı')
plt.show()

sns.boxplot(x='Cinsiyet', y='Yazma', data=df)
plt.title('Cinsiyete Göre Yazma Başarısı')
plt.show()
from sklearn.linear_model import LinearRegression
import numpy as np

# Özel ders alan ve almayan öğrencilerin ortalama başarı puanları
mean_scores = df.groupby('Ozel Ders')[['Matematik', 'Okuma', 'Yazma']].mean()
print(mean_scores)

# Regresyon modeli
X = pd.get_dummies(df[['Cinsiyet', 'Ebeveyn Egitim Seviyesi', 'Okul Yemekhanesi', 'Ozel Ders']], drop_first=True)
y = df[['Matematik', 'Okuma', 'Yazma']]

model = LinearRegression()
model.fit(X, y)

# Modelin katsayıları
print(model.coef_)
print(model.intercept_)
# Korelasyon matrisi
corr_matrix = df.corr()
print(corr_matrix)

# Korelasyon ısı haritası
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Korelasyon Matrisi')
plt.show()
# Okuma ve diğer değişkenler arasındaki ilişki
sns.pairplot(df, x_vars=['Okuma'], y_vars=['Matematik', 'Yazma'], kind='reg')
plt.show()

# Çoklu regresyon modeli
X = df[['Okuma']]
y_math = df['Matematik']
y_writing = df['Yazma']

model_math = LinearRegression()
model_math.fit(X, y_math)

model_writing = LinearRegression()
model_writing.fit(X, y_writing)

print(f"Okuma ve Matematik arasındaki ilişki: {model_math.coef_[0]}")
print(f"Okuma ve Yazma arasındaki ilişki: {model_writing.coef_[0]}")
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Karar ağacı ve random forest modelleri
tree_model = DecisionTreeRegressor()
forest_model = RandomForestRegressor()

tree_model.fit(X, y_math)
forest_model.fit(X, y_math)

tree_pred = tree_model.predict(X)
forest_pred = forest_model.predict(X)

print(f"Decision Tree MSE: {mean_squared_error(y_math, tree_pred)}")
print(f"Random Forest MSE: {mean_squared_error(y_math, forest_pred)}")
