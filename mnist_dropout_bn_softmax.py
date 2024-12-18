import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 加载MNIST图像数据
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 从缓冲区读取数据，并根据偏移量解析数据
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 重新整形数据以匹配图像尺寸
        data = data.reshape(-1, 28, 28)
    return data

# 加载MNIST标签数据
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 对标签进行独热编码（One-Hot Encoding）
def to_one_hot(labels, num_classes=10):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1
    return one_hot_labels 

# 初始化神经网络参数
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}#用于存储所有层的参数
    L = len(layer_dims)#获取网络的层数
    for l in range(1, L):
        # 使用He初始化方法初始化权重矩阵，偏置初始化为零
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        # 初始化 Batch Norm 参数 gamma全部初始化为1  beta全部初始化为0
        parameters['gamma' + str(l)] = np.ones((layer_dims[l], 1))
        parameters['beta' + str(l)] = np.zeros((layer_dims[l], 1))
    #这个函数返回一个包含所有层参数的字典 parameters，其中每一层的参数
    #都以 W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, ... 的形式存储
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# 计算交叉熵损失函数
def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = -1 / m * np.sum(np.multiply(Y.T, np.log(AL + 1e-8)) + np.multiply(1 - Y.T, np.log(1 - AL + 1e-8)))
    return np.squeeze(cost)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# bn批量归一
def batch_normalize(Z, epsilon=1e-8):
    mu = np.mean(Z, axis=1, keepdims=True)
    var = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mu) / np.sqrt(var + epsilon)
    return Z_norm, mu, var

def batch_norm_backward(dZ_norm, Z_norm, mu, var, epsilon=1e-8):
    m = dZ_norm.shape[1]
    dvar = np.sum(dZ_norm * (Z_norm - mu), axis=1, keepdims=True) * -0.5 * (var + epsilon) ** (-1.5)
    dmu = np.sum(dZ_norm * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + dvar * np.mean(-2 * (Z_norm - mu), axis=1, keepdims=True)
    dZ = dZ_norm / np.sqrt(var + epsilon) + dvar * 2 * (Z_norm - mu) / m + dmu / m
    return dZ

"""
单独的dropout步骤
生成与A相同形状的掩码矩阵D，根据keep_prob的概率随机生成0，1
A与D相乘并除以keep_prob得到新的激活值A
"""
def dropout_forward(A, keep_prob):
    D = np.random.rand(*A.shape)
    D = (D < keep_prob).astype(int)
    A = A * D / keep_prob
    return A, D

# dropout反向传递
def dropout_backward(dA, D, keep_prob):
    dA = dA * D / keep_prob
    return dA

"""
用于整个前向传播过程，包括多个隐藏层计算
每个隐藏层激活后调用dropout_forward应用dropout操作
得到掩码矩阵D存下来以备反向传播时使用
输出层使用softmax激活
输出AL与包含缓存D的列表
"""
def forward_propagation_with_dropout(X, parameters, keep_prob):
    caches = []
    A = X.T
    L = len(parameters) // 4  # W, b, gamma, beta

    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        # Batch Normalization
        Z_norm, mu, var = batch_normalize(Z)
        Z_tilde = parameters['gamma' + str(l)] * Z_norm + parameters['beta' + str(l)]
        A = relu(Z_tilde)

        # Dropout
        if l < L - 1:  # 不对输出层进行 Dropout
            A, D = dropout_forward(A, keep_prob)
            caches.append((A_prev, Z, Z_norm, mu, var, D))
        else:
            caches.append((A_prev, Z, Z_norm, mu, var))

    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = softmax(ZL)
    caches.append((A, ZL))

    return AL, caches

# 带有dropout的反向传播
def backward_propagation_with_dropout(AL, Y, caches, parameters, keep_prob):
    grads = {}
    L = len(caches) # 网络层数
    m = AL.shape[1] # 训练样本数量
    Y = Y.T # 真实标签的独热编码矩阵
    dZL = AL - Y # 输出层的激活值相对于损失的梯度

    # 输出层的反向传播
    A_prev, ZL = caches[L - 1]
    dWL = 1/m * np.dot(dZL, A_prev.T) #输出层权重
    dbL = 1/m * np.sum(dZL, axis=1, keepdims=True) #偏置
    dA_prev = np.dot(parameters['W' + str(L)].T, dZL)

    grads['dW' + str(L)] = dWL
    grads['db' + str(L)] = dbL

    # 隐藏层的反向传播
    for l in reversed(range(L - 1)): # 从倒数第二层开始
        cache = caches[l]
        A_prev, Z, Z_norm, mu, var = cache[:5]
        if len(cache) == 6:
            D = cache[5]
            dA_prev = dropout_backward(dA_prev, D, keep_prob)

        # Batch Norm 反向传播
        dZ_tilde = relu_backward(dA_prev, Z)
        dZ_norm = dZ_tilde * parameters['gamma' + str(l + 1)]
        dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)
        dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
        dZ = batch_norm_backward(dZ_norm, Z_norm, mu, var)
        grads['dW' + str(l + 1)] = 1 / m * np.dot(dZ, A_prev.T)
        grads['db' + str(l + 1)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        grads['dgamma' + str(l + 1)] = dgamma
        grads['dbeta' + str(l + 1)] = dbeta
        dA_prev = np.dot(parameters['W' + str(l + 1)].T, dZ)

    return grads #返回梯度

# 梯度下降更新参数
def update_parameters(parameters, grads, learning_rate, cache, epsilon=1e-8):
    L = len(parameters) // 4  # W, b, gamma, beta

    for l in range(1, L + 1):
        # 累积梯度平方
        cache['dW' + str(l)] += grads['dW' + str(l)] ** 2
        cache['db' + str(l)] += grads['db' + str(l)] ** 2
        cache['dgamma' + str(l)] += grads.get('dgamma' + str(l), 0) ** 2
        cache['dbeta' + str(l)] += grads.get('dbeta' + str(l), 0) ** 2
        # 参数更新
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)] / (np.sqrt(cache['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)] / (np.sqrt(cache['db' + str(l)]) + epsilon)
        parameters['gamma' + str(l)] -= learning_rate * grads.get('dgamma' + str(l), 0) / (np.sqrt(cache['dgamma' + str(l)]) + epsilon)
        parameters['beta' + str(l)] -= learning_rate * grads.get('dbeta' + str(l), 0) / (np.sqrt(cache['dbeta' + str(l)]) + epsilon)

    return parameters

def predict(X, parameters):
    AL, _ = forward_propagation_with_dropout(X, parameters, keep_prob)
    return AL.T  # 返回转置后的预测值

# 神经网络模型函数
def neural_network_model_with_dropout(X, Y, layers_dims, learning_rate, num_epochs, batch_size, keep_prob):
    parameters = initialize_parameters(layers_dims)
    costs = []
    train_accuracy_array = []

    # 初始化 cache
    cache = {'dW' + str(l): 0 for l in range(1, len(layers_dims))}
    cache.update({'db' + str(l): 0 for l in range(1, len(layers_dims))})
    cache.update({'dgamma' + str(l): 0 for l in range(1, len(layers_dims))})
    cache.update({'dbeta' + str(l): 0 for l in range(1, len(layers_dims))})

    for epoch in range(num_epochs):
        epoch_cost = 0
        num_minibatches = X.shape[0] // batch_size

        for i in range(num_minibatches):
            start = i * batch_size
            end = min(start + batch_size, X.shape[0])
            X_batch = X[start:end]
            Y_batch = Y[start:end]  

            # 前向传播（带 Dropout）
            AL, caches = forward_propagation_with_dropout(X_batch, parameters, keep_prob)
            epoch_cost += compute_cost(AL, Y_batch)

            # 反向传播（带 Dropout）
            grads = backward_propagation_with_dropout(AL, Y_batch, caches, parameters, keep_prob)  
            
            # 参数更新
            parameters = update_parameters(parameters, grads, learning_rate, cache)

        epoch_cost /= num_minibatches
        costs.append(epoch_cost) 

        train_predictions = predict(train_images, parameters)
        train_predictions_labels = train_predictions.argmax(axis=1)
        train_true_labels = train_labels.argmax(axis=1)
        train_accuracy = np.mean(train_predictions_labels == train_true_labels) * 100
        train_accuracy_array.append(train_accuracy)

        if epoch % 1 == 0:
            print(f"Cost after epoch {epoch}/{num_epochs}: {epoch_cost}\n,train_accuracy:{train_accuracy:.2f}%")
            print("-" * 40)  # 分隔线

    test_predictions = predict(test_images, parameters)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    # print("predicted_labels:\n")
    # print(len(predicted_labels))
    # print("\n")
    # print(len(test_labels))
    test_true_labels = test_labels.argmax(axis=1)
    # print("true_labels:\n")
    # print(len(true_labels))

    test_accuracy = np.mean(test_predicted_labels == test_true_labels) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("混淆矩阵如下：\n")
    print(confusion_matrix(test_predicted_labels, test_true_labels))

    return parameters, costs, train_accuracy_array

learning_rate = 0.085
num_epochs = 400
batch_size = 64
keep_prob = 0.4

# 设置本地文件路径
train_images_path = 'mnist/train-images-idx3-ubyte.gz'
train_labels_path = 'mnist/train-labels-idx1-ubyte.gz'
test_images_path = 'mnist/t10k-images-idx3-ubyte.gz'
test_labels_path = 'mnist/t10k-labels-idx1-ubyte.gz'

# 加载训练集和测试集
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# 数据预处理
train_labels = to_one_hot(train_labels)
test_labels = to_one_hot(test_labels)

# 将图像数据转换为一维向量，并进行归一化
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 调用神经网络模型
layers_dims = [784, 384, 256, 10]
trained_parameters, costs, train_accuracy = neural_network_model_with_dropout(train_images, train_labels, layers_dims, learning_rate, num_epochs, batch_size, keep_prob)

plt.figure(figsize=(5, 5))
plt.plot(range(num_epochs), costs, label = 'Training Loss', color = 'blue')
plt.xlabel("Epoches")
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(range(num_epochs), train_accuracy, label = 'Training Acc', color = 'black')
plt.xlabel("Epoches")
plt.ylabel('Loss')
plt.title('Training Acc per Epoch')
plt.legend()
plt.tight_layout()
plt.show()


