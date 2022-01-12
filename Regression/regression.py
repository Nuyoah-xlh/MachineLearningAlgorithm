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


################################# softmax回归 start ###################################

# softmax函数
def softmax(z):
    # 减去x 的最大值以获得数值稳定性
    z-=np.max(z)
    denominator = np.sum(np.exp(z))
    softmax = np.exp(z) / denominator
    return softmax

# softmax的交叉熵损失函数
def cross_entropy_loss_softmax(one_hot_mat,softmax1):
    temp=np.multiply(one_hot_mat, np.log(softmax1))
    return -(1.0/one_hot_mat.shape[0])*np.sum(temp)

# 批量梯度下降-softmax
def gradient_down_softmax(X,Y,alpha,count,n_classes):
    # 转为矩阵
    X = np.mat(X)
    Y = np.mat(Y)
    # 初始化thetas
    thetas=np.zeros((n_classes,X.shape[1]))
    # 定义损失函数值
    loss=0
    # 计数变量
    cur=0
    while cur<count:
        cur+=1
        softmax1=softmax(np.dot(X,thetas.T))
        one_hot_mat=one_hot(Y,X,n_classes)
        # 损失函数
        loss=cross_entropy_loss_softmax(one_hot_mat,softmax1)
        dw = (1.0 / len(X)) * np.dot(X.T,one_hot_mat-softmax1)
        # 更新thetas
        thetas = thetas + alpha *dw.T
        print("softmax回归-第" + cur.__str__() + "次迭代   损失函数值：" + loss.__str__())
    return thetas,loss

# 随机梯度下降-softmax回归
def SGD_softmax(X,Y,alpha,count,n_classes):
    # 转为矩阵
    X = np.mat(X)
    Y = np.mat(Y)
    # 初始化thetas
    thetas=np.zeros((n_classes,X.shape[1]))
    # 定义损失函数值
    loss=0
    # 计数变量
    cur=0
    while cur<count:
        cur+=1
        # 找一个随机样本
        index = np.random.randint(1, X.shape[0])
        x = X[index, :]
        y = Y[index, 0]
        softmax1=softmax(np.dot(x,thetas.T))
        one_hot_mat=one_hot(y,x,n_classes)
        # 损失函数
        loss=cross_entropy_loss_softmax(one_hot(Y,X,n_classes),softmax(np.dot(X,thetas.T)))
        dw =np.dot(x.T,one_hot_mat-softmax1)
        # 更新thetas
        thetas = thetas + alpha/len(X) *dw.T
        print("softmax回归-第" + cur.__str__() + "次迭代   损失函数值：" + loss.__str__())
    return thetas,loss

# 求得y的one-hot矩阵
def one_hot(Y, X, n_classes):
    one_hot = np.zeros((len(X), n_classes))
    one_hot[np.arange(len(X)), Y.T.astype('int64')] = 1
    return np.mat(one_hot)

################################# softmax回归 end ###################################

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


#################softmax回归###############
alpha2=0.00001
count2=1000
n_classes=2
thetas2,loss2=SGD_softmax(dataset_x,dataset_y,alpha2,count2,n_classes)


##########输出二者的最终结果########
print("############################")

print("逻辑回归结果：")
print("thetas参数: "+thetas1.__str__())
print("交叉熵损失："+loss.__str__())
print("############################")

print("softmax回归结果：")
print("thetas参数："+thetas2.__str__())
print("交叉熵损失："+loss2.__str__())

print("############################")


