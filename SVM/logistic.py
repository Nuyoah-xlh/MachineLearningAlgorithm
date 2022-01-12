import pandas as pd
import numpy as np


# 数据格式转换
def dataset_format(dataset):
    ans=[]
    # 将被空格分隔开的数据提取出来，并统一存储后返回
    for str in dataset:
        temp=str.split()
        for i in range(len(temp)):
            temp[i]=float(temp[i])
        ans.append(temp)
    return ans

################################# logistic回归 start ###################################

#sigmoid函数
def sigmoid(z):
    s = 1.0/(1.0+np.exp(-z))
    return s


# 逻辑回归的交叉熵损失函数（这里越小越好）
# thetas为参数，X为特征值，Y为y值
def cross_entropy_loss_logistic(thetas,X,Y):
    sig = sigmoid(np.dot(X, thetas))
    # 这里np.log()函数里加入一个固定值1e-10,保证浮点数有效，可以有效防止取对数溢出的情况
    J=np.multiply(-Y,np.log(sig+1e-10))+np.multiply(Y-1,np.log(1-sig+1e-10))
    return np.sum(J)/float(len(X))


# 批量梯度下降-逻辑回归
def gradient_down_logistic(thetas,X,Y,alpha,count):
    # 将X,y转为矩阵运算
    X=np.mat(X)
    Y=np.mat(Y)
    # 迭代指定次数
    cur=0
    # 初始的loss
    loss = cross_entropy_loss_logistic(thetas, X, Y)
    while cur<count:
        # 计算交叉熵损失函数
        loss = cross_entropy_loss_logistic(thetas,X,Y)
        # 计算sigmoid及与y的差值
        temp = sigmoid(np.dot(X, thetas))-Y
        gradient = (1.0 /len(X)) * np.dot(X.T, temp)
        # 更新thetas值
        thetas = thetas- alpha*gradient
        # 迭代次数加一
        cur+=1
        print("逻辑回归-第"+cur.__str__()+"次迭代   交叉熵损失函数值："+loss.__str__())
    # 返回thetas和交叉熵损失
    return thetas,loss

# 随机梯度下降-逻辑回归
def SGD_logistic(thetas,X,Y,alpha,count):
    # 将X,y转为矩阵运算
    X = np.mat(X)
    Y = np.mat(Y)
    # 迭代次数计数
    cur = 0
    # 初始的loss
    loss = cross_entropy_loss_logistic(thetas, X, Y)
    while cur < count:
        # 计算交叉熵损失函数
        loss = cross_entropy_loss_logistic(thetas, X, Y)
        # 找一个随机样本
        index=np.random.randint(1,X.shape[0])
        x=X[index,:]
        y=Y[index,0]
        # 计算sigmoid及与y的差值
        temp = sigmoid(np.dot(x, thetas)) - y
        gradient =  np.dot(x.T, temp)
        # 更新thetas值
        thetas = thetas - alpha * gradient
        # 迭代次数加一
        cur += 1
        print("逻辑回归-第" + cur.__str__() + "次迭代   交叉熵损失函数值：" + loss.__str__())
    # 返回thetas和交叉熵损失
    return thetas, loss



################################# logistic回归 end ###################################




################################# 函数调用执行 ##############################################
# 提取数据集的x特征
dataset_x=pd.read_csv("ex4x.dat", header=None)[0]
# 格式化数据
dataset_x=dataset_format(np.array(dataset_x))
# 在第一列新增一列，值均为1.0，即设置x0=1.0
dataset_x = np.column_stack(([1 for i in range(len(dataset_x))],dataset_x))
# 转为DataFrame数据类型,便于后续操作
dataset_x=pd.DataFrame(dataset_x)


# 提取数据集的y值
dataset_y=pd.read_csv("ex4y.dat", header=None)[0]
# 格式化数据
dataset_y=pd.DataFrame(dataset_y)

##############逻辑回归###################
# 设置学习率
alpha1=0.001
# 设置迭代次数
count1=1000
# 设置thetas参数，初始均为0
thetas1=np.ones((dataset_x.shape[1], 1))
# 梯度下降求得最佳的thetas和最小的交叉熵损失
thetas1,loss=SGD_logistic(thetas1,dataset_x,dataset_y,alpha1,count1)


##########输出二者的最终结果########
print("############################")

print("逻辑回归结果：")
print("thetas参数: "+thetas1.__str__())
print("交叉熵损失："+loss.__str__())
print("############################")

print("############################")


