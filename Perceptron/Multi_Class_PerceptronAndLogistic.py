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


# 求得y的one-hot矩阵
def one_hot(Y, X, n_classes):
    one_hot = np.zeros((len(X), n_classes))
    one_hot[np.arange(len(X)), Y.T.astype('int64')] = 1
    return np.mat(one_hot)

################################# 多类感知机算法 start ###################################
# 随机梯度下降-感知机算法
def SGD_perceptron(X,Y,alpha,count,n_classes):
    # 将X,y转为矩阵运算
    X = np.mat(X)
    Y = np.mat(Y)
    # 初始化参数矩阵
    thetas = np.random.random((n_classes, X.shape[1]))
    # 控制迭代次数
    cur=0
    # 记录损失函数
    loss = 0
    # 进行迭代
    while cur<count:
        # 迭代次数+1
        cur+=1
        # C记录最大值
        C=np.zeros((1,X.shape[0])).T
        # C_index记录最大值对应的类别
        C_index = np.zeros((1, X.shape[0])).T
        # 损失函数初始化
        loss=0
        for i in range(X.shape[0]):
            # max_c记录最大值，max_c_index记录最大值对应的下标
            max_c=-1
            max_c_index=-1
            # 取可能性概率最大的
            for j in range(n_classes):
                if float(np.dot(X[i,:],thetas[j,:].T))>max_c:
                    max_c_index=j
                    max_c=float(np.dot(X[i,:],thetas[j,:].T))
            C[i,0]=max_c
            C_index[i,0]=max_c_index
            # thetas参数更新规则
            for j in range(n_classes):
                ck=C_index[i,0]
                yk=Y[i,0]
                if j==ck and j!=yk:
                    thetas[j,:]=thetas[j,:]-alpha*X[i,:]
                elif j==yk and j!=ck:
                    thetas[j,:]=thetas[j,:]+alpha*X[i,:]
            # 累加代价函数值
            loss+=abs(C[i,0]-np.dot(X[i,:],thetas[int(Y[i,0]),:].T))
        print("多类感知机-第" + cur.__str__() + "次迭代   代价函数值：" + float(loss).__str__())
    return thetas,loss
################################# 多类感知机算法 end ###################################


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


# 随机梯度下降-softmax回归
def SGD_softmax(X,Y,alpha,count,n_classes):
    # 转为矩阵
    X = np.mat(X)
    Y = np.mat(Y)
    # 初始化thetas
    thetas=np.random.random((n_classes,X.shape[1]))
    # 定义损失函数值
    loss=0
    # 计数变量
    cur=0
    while cur<count:
        cur+=1
        for i in range(X.shape[0]):
            # 找一个随机样本
            index = np.random.randint(1, X.shape[0])
            x = X[index, :]
            y = Y[index, 0]
            # 计算softmax
            softmax1=softmax(np.dot(x,thetas.T))
            one_hot_mat=one_hot(y,x,n_classes)
            dw =np.dot(x.T,one_hot_mat-softmax1)
            # 更新thetas
            thetas = thetas + alpha/len(X) *dw.T
        # 损失函数
        loss = cross_entropy_loss_softmax(one_hot(Y, X, n_classes), softmax(np.dot(X, thetas.T)))
        print("softmax回归-第" + cur.__str__() + "次迭代   损失函数值：" + loss.__str__())
    return thetas,loss

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


################# 感知机回归 ###############
alpha1=0.00001
count1=1000
n_classes1=2
thetas1,loss1=SGD_perceptron(dataset_x,dataset_y,alpha1,count1,2)


#################softmax回归###############
alpha2=0.00001
count2=2000
n_classes2=2
thetas2,loss2=SGD_softmax(dataset_x,dataset_y,alpha2,count2,n_classes2)


##########输出二者的最终结果########

print("############################")

print("多类感知机结果：")
print("thetas（包括w,b）参数："+thetas1.__str__())
print("错误分类损失："+loss1.__str__())

print("############################")


print("############################")

print("softmax回归结果：")
print("thetas参数："+thetas2.__str__())
print("交叉熵损失："+loss2.__str__())

print("############################")


