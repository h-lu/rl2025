import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 使用 keras 加载MNIST数据集
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 从本地加载数据集
x_train = np.load('mnist/data/x_train.npy')
y_train = np.load('mnist/data/y_train.npy')
x_test = np.load('mnist/data/x_test.npy')
y_test = np.load('mnist/data/y_test.npy')

# 查看数据集形状
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# 绘制前10张手写数字图片
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off') # 关闭坐标轴
plt.tight_layout() # 调整子图，防止重叠
plt.show()


# 将 x_train 变换为 60000*784 的矩阵
x_train = x_train.reshape((x_train.shape[0], 28*28))

# 将 x_test 变换为 10000*784 的矩阵
x_test = x_test.reshape((x_test.shape[0], 28*28))

x_train.shape

# 将 y_train, y_test 变为哑变量形式
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 构造随机森林模型
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=20, max_samples=0.2, n_jobs=-1)

# 训练模型
rf_classifier.fit(x_train, y_train)

# 预测
y_pred = rf_classifier.predict(x_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f"准确率: {accuracy}")

# 找到预测错误的图片索引
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"预测错误的图片数量: {len(misclassified_indices)}")

# 绘制前10个预测错误的图片
plt.figure(figsize=(10, 5))
for i in range(min(10, len(misclassified_indices))):
    index = misclassified_indices[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"真实值: {y_test[index]}, 预测值: {y_pred[index]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 找到预测正确的图片索引
classified_indices = np.where(y_pred == y_test)[0]
print(f"预测正确的图片数量: {len(classified_indices)}")

# 绘制前10个预测正确的图片
plt.figure(figsize=(10, 5))
for i in range(min(10, len(classified_indices))):
    index = classified_indices[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"真实值: {y_test[index]}, 预测值: {y_pred[index]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

### 使用 keras 创建稠密连接(全连接)网络模型
from keras.models import Sequential
from keras.layers import Dense, Input

# 创建顺序模型
model = Sequential()
model.add(Dense(units=512, input_shape=(784,), activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()
# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型并记录历史
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 早停机制
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# 学习率衰减
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])


# 测试集上的准确率
y_pred = model.predict(x_test)
accuracy = np.mean(y_pred == y_test)
print(f"准确率: {accuracy}")

# 将 y 变换为数值形式
y_pred_classes = np.argmax(y_pred, axis=1)
y_classes = np.argmax(y_test, axis=1)


# 找到预测错误的图片索引
misclassified_indices = np.where(y_pred_classes != y_classes)[0]
print(f"预测错误的图片数量: {len(misclassified_indices)}")

# 绘制前10个预测错误的图片
plt.figure(figsize=(10, 5))
for i in range(min(10, len(misclassified_indices))):
    index = misclassified_indices[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"真实值: {y_classes[index]}, 预测值: {y_pred_classes[index]}")
    plt.axis('off')
plt.tight_layout()
plt.show()












