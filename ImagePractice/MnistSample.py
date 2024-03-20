import tensorflow as tf

(mnist_x, mnist_y), _ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)

# 화면 출력
print(mnist_y[0:10])

import matplotlib.pyplot as plt
plt.imshow(mnist_x[4], cmap='gray')
plt.show()
