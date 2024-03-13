import os
import tensorflow as tf
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = 'https://raw.githubusercontents.com/blackdew/tensorflow1/master/csv/iris.csv'
storage_options = {'User-Agent': 'Mozilla/5.0'}

dataset = pd.get_dummies(pd.read_csv(path, storage_options = storage_options))

# 독립변수
x = dataset[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
# 종속변수
y = dataset[['품종_setosa', '품종_versicolor', '품종_virginica']]

# ## 모델의 구조 정의
Input = tf.keras.layers.Input(shape=(4,))
Layer2 = tf.keras.layers.Dense(10, activation='swish')(Input)
Layer3 = tf.keras.layers.Dense(10, activation='swish')(Layer2)
Output = tf.keras.layers.Dense(3, activation='softmax')(Layer3)
model = tf.keras.models.Model(Input, Output)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

## 학습
print("학습 시작")
model.fit(x, y, epochs=2000, verbose=0)
print("학습 완료")

print(model.predict(x[:5]))
print(y[:5])
