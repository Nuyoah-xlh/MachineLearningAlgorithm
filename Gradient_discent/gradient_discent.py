import numpy as np
import pandas as pd


# 将字符串形式的数据转为浮点数，并返回
# flag=0表示返回特征x，flag=1表示返回y值
def getData(data, flag):
    str = data.split()
    ans = []
    for s in str:
        ans.append(float(s))
    if flag == 0:
        # 添加一个x0=1.0便于后面计算
        return [1.0] + ans[:13]
    else:
        return ans[13]


# 计算损失函数（平方误差损失函数,使用平均值）
def compute_loss(Thetas):
    T1 = np.mat(Thetas).T
    loss = np.dot((np.dot(X1, T1) - Y1).T, (np.dot(X1, T1) - Y1))
    return float(loss) / 2 / float(len(trainX))


# 进行梯度下降（第一个公式-标量乘法）
def gradient_discent_scalar(alpha, Thetas):
    # 样本数量
    X_len = len(trainX)
    # 初始loss值
    loss_temp = compute_loss(Thetas)
    while True:
        dz = 0
        # 临时存储变量
        dk = [0 for _ in range(len(Thetas))]
        # 取每个数据样本
        for j in range(X_len):
            z = 0
            # 取每个参数与对应特征相乘
            for n in range(len(Thetas)):
                z += Thetas[n] * trainX[j][n]
            # 相乘并累加后，减去yj
            dz = z - trainY[j]
            # 记下每个xn与之相乘的结果
            for n in range(len(Thetas)):
                dk[n] += trainX[j][n] * dz / float(X_len)
        for n in range(len(Thetas)):
            # lr为学习率
            Thetas[n] -= alpha * dk[n]
        # 当损失函数基本不再下降时，退出循环
        current = compute_loss(Thetas)
        if abs(loss_temp - current) <= 1e-5:
            break
        # 存储这次的损失函数值
        loss_temp = current
        print("公式1-损失函数值loss:" + str(loss_temp))
    return Thetas, loss_temp


# 进行梯度下降（第二个公式-向量乘法）
def gradient_discent_vector(alpha, Thetas):
    # 初始loss值
    loss_temp = compute_loss(Thetas)
    while True:
        T1 = np.mat(Thetas)
        temp = [0.0 for i in range(14)]
        temp = np.mat(temp)
        for i in range(len(trainX)):
            Xi = trainX[i]
            temp = temp + ((np.dot(Xi, T1.T) - trainY[i]) * Xi)
        Thetas = T1 - alpha / float(len(trainX)) * temp
        current = compute_loss(Thetas)
        # 当损失函数基本不再下降时，退出循环
        if abs(loss_temp - current) <= 1e-5:
            break
        # # 存储这次的损失函数值
        loss_temp = current
        print("公式2-损失函数值loss:" + str(loss_temp))
    return Thetas, loss_temp


# 进行梯度下降（第三种方式-矩阵乘法）
def gradient_discent_matrix(alpha, Thetas):
    # 初始损失函数
    loss_temp = compute_loss(Thetas)
    while True:
        T1 = np.mat(Thetas)
        temp = alpha / float(len(trainX)) * np.dot(X1.T, np.dot(X1, T1.T) - Y1)
        Thetas = T1 - temp.T
        current = compute_loss(np.array(Thetas)[0])
        # 当损失函数基本不再下降时，退出循环
        if abs(loss_temp - current) <= 1e-5:
            break
        # 存储这次的损失函数值
        loss_temp = current
        print("公式3-损失函数值loss:" + str(loss_temp))
    return Thetas, loss_temp


# 计算平均平方误差
def predict(Thetas):
    T1 = np.mat(Thetas).T
    error = np.dot((np.dot(X2, T1) - Y2).T, (np.dot(X2, T1) - Y2))
    return float(error) / float(len(testX))


# 读取数据集
dataset = pd.read_csv("housing.data", sep="\t", header=None)
dataset = dataset[0]
# 分割为训练集和测试集
dataset1 = dataset[:400]
dataset2 = dataset[400:]
# 处理数据格式
# 存储每个样本的特征
trainX = []
# 存储每个样本的y值
trainY = []
# 存储每个测试数据的特征
testX = []
# 存储每个测试数据的y值
testY = []
# 处理数据
for data in dataset1:
    trainX.append(getData(data, 0))
    trainY.append(getData(data, 1))
for data in dataset2:
    testX.append(getData(data, 0))
    testY.append(getData(data, 1))
# 将特征数据和y值数据转为矩阵便于运算
X1 = np.mat(trainX)
Y1 = np.mat(trainY).T
X2 = np.mat(testX)
Y2 = np.mat(testY).T

# thetas参数初始化为[0,1]之间的随机浮点数，共14个参数，X特征在数据处理时已添加了一个x0=1.0，共14个
thetas1 = np.random.rand(len(trainX[0]))
thetas2 = np.random.rand(len(trainX[0]))
thetas3 = np.random.rand(len(trainX[0]))
# 学习率经过测试，选择0.00000715效果最佳
alpha = 0.00000715

################## 函数执行开始 ##############################

#####第一种方式（大约需要20min,降低gradient_discent_scalar的终止条件可减少时间）#####
print("第一种方法开始运行：")
thetas1, loss1 = gradient_discent_scalar(alpha, thetas1)
print("第一种方法运行结束")

#####第二种方式（大约需要30min,降低gradient_discent_vector的终止条件可减少时间）#####
print("第二种方法开始运行：")
thetas2, loss2 = gradient_discent_vector(alpha, thetas2)
print("第二种方法运行结束")

#####第三种方式（大约需要30s,降低gradient_discent_vector的终止条件可减少时间）#####
print("第三种方法开始运行：")
thetas3, loss3 = gradient_discent_matrix(alpha, thetas3)
print("第三种方法运行结束")

print("第一种方式运行结果：")
print("损失函数值loss:" + str(loss1))
print("thetas参数值：" + str(thetas1))
print("平均平方误差:" + str(predict(thetas1)))

print("\n\n第二种方式运行结果：")
print("损失函数值loss:" + str(loss2))
print("thetas参数值：" + str(np.array(thetas2)[0]))
print("平均平方误差:" + str(predict(thetas2)))

print("\n\n第三种方式运行结果：")
print("损失函数值loss:" + str(loss3))
print("thetas参数值：" + str(np.array(thetas3)[0]))
print("平均平方误差:" + str(predict(thetas3)))
