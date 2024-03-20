import os
import tensorflow as tf
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = 'https://raw.githubusercontents.com/blackdew/tensorflow1/master/csv/boston.csv'
storage_options = {'User-Agent': 'Mozilla/5.0'}

dataset = pd.read_csv(path, storage_options = storage_options)

print(dataset.columns)

# 독립변수
x = dataset[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
# 종속변수
y = dataset[['medv']]

## 모델의 구조 정의
Input = tf.keras.layers.Input(shape=(13,))

# 히든 레이어 추가
Layer = tf.keras.layers.Dense(10, activation='sigmoid')(Input)

Output = tf.keras.layers.Dense(1)(Input)



model = tf.keras.models.Model(Input, Output)
model.compile(loss='mse')

## 학습
print("학습 시작")
model.fit(x, y, epochs=2000, verbose=0)
print("학습 완료")

print(model.predict(x[:5]))
print(y[:5])
