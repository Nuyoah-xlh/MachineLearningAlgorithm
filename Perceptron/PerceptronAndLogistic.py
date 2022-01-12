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

################################# 感知机算法 start ###################################

# 感知机的假设函数，x>=0返回1，否则返回0
def hw(x):
    if x>=0:
        return 1
    else:
        return 0

# 随机梯度下降-感知机算法
def SGD_perceptron(X,Y,alpha,count):
    # 将X,y转为矩阵运算
    X = np.mat(X)
    Y = np.mat(Y)
    # 初始化参数矩阵
    thetas=np.random.rand(3)
    thetas=np.mat(thetas)
    # 控制迭代次数
    cur=0
    # 记录损失函数
    loss = 0
    # 进行迭代
    while cur<count:
        # 迭代次数+1
        cur+=1
        # 损失函数初始化
        loss=0
        # 随机梯度下降策略
        for i in range(X.shape[0]):
            # 计算其hw值
            h=hw(float(np.dot(X[i,:],thetas.T)))
            # 计算损失函数
            loss+=(h-float(Y[i,0]))*float(np.dot(X[i,:],thetas.T))
            # 更新参数  误差*输入
            thetas+=alpha*(float(Y[i,0])-h)*X[i,:]
        print("感知机-第" + cur.__str__() + "次迭代   错误分类损失函数值：" + loss.__str__())
    return thetas,loss


################################# 感知机算法 end ###################################




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
        for i in range(X.shape[0]):
            # 计算交叉熵损失函数
            loss = cross_entropy_loss_logistic(thetas, X, Y)
            # 找一个随机样本
            index=np.random.randint(0,X.shape[0])
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

############## 感知机算法 ###################
# 设置学习率
alpha1=0.00001
# 迭代次数
count1=1000
thetas1,loss1=SGD_perceptron(dataset_x,dataset_y,alpha1,count1)


############## 逻辑回归 ###################
# 设置学习率
alpha2=0.001
# 设置迭代次数
count2=1000
# 设置thetas参数，初始均为0
thetas2=np.random.random((dataset_x.shape[1],1))
# 梯度下降求得最佳的thetas和最小的交叉熵损失
thetas2,loss2=SGD_logistic(thetas2,dataset_x,dataset_y,alpha2,count2)


print("############################")
print("感知机结果：")
print("thetas参数（包括w,b）: "+thetas1.__str__())
print("错误分类损失："+loss1.__str__())

print("############################")



########## 输出二者的最终结果 ########
print("############################")
print("逻辑回归结果：")
print("thetas参数: "+thetas2.__str__())
print("交叉熵损失："+loss2.__str__())
print("############################")



