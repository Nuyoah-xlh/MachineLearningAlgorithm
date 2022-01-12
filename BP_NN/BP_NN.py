import numpy as np
import pandas as pd
import random


############################################### 数据获取及格式处理 start ##############################
# 加载数据集并进行格式处理
def load_data():
    # 提取数据集的x特征
    dataset_x = pd.read_csv("ex4x.dat", header=None)[0]
    # 格式化数据
    dataset_x = dataset_format(np.array(dataset_x))
    # 转为DataFrame数据类型,便于后续操作
    dataset_x = list(dataset_x)
    # 提取数据集的y值
    dataset_y = pd.read_csv("ex4y.dat", header=None)[0]
    dataset_y = list(dataset_y)
    # 拼接x,y值到dataset_x中
    for i in range(len(dataset_y)):
        dataset_x[i].append(float(dataset_y[i]))
    # 转换为np数组
    dataset_x=np.array(dataset_x)
    # 将所有训练数据打乱，使之无序
    random.shuffle(dataset_x)
    # 将训练集平均分为5份，便于后续5倍交叉验证
    train1=get_train_data(dataset_x[0:64])
    test1=get_test_data(dataset_x[64:80])
    train2 = get_train_data(dataset_x[-16:48])
    test2 = get_test_data(dataset_x[48: 64])
    train3 = get_train_data(dataset_x[-32:32])
    test3 = get_test_data(dataset_x[32:48])
    train4 = get_train_data(dataset_x[-48:16])
    test4 = get_test_data(dataset_x[16:32])
    train5 = get_train_data(dataset_x[16:])
    test5 = get_test_data(dataset_x[:16])
    # 将每次的训练和测试数据对应，并返回
    # train_data_t存储每次验证时所需要的所有训练数据
    train_data_t=[]
    train_data_t.append(train1)
    train_data_t.append(train2)
    train_data_t.append(train3)
    train_data_t.append(train4)
    train_data_t.append(train5)
    # test_data_t存储每次验证时所需要的所有训练数据
    test_data_t = []
    test_data_t.append(test1)
    test_data_t.append(test2)
    test_data_t.append(test3)
    test_data_t.append(test4)
    test_data_t.append(test5)
    # 返回
    return train_data_t,test_data_t

# 获取训练集，并进行数据的格式处理
def get_train_data(train_data):
    train_x , train_y = train_data[:, :-1], train_data[:, -1]
    train_x_r = [x.reshape(len(x), 1) for x in train_x]
    # 构造one-hot矩阵
    train_y_r = [np.array([[1 if y==0 else 0],
                           [1 if y==1 else 0]])
                           for y in train_y]
    train_data_t = [[x, y] for x, y in zip(train_x_r, train_y_r)]
    return train_data_t

# 获取测试集并进行数据格式的处理
def get_test_data(test_data):
    test_x , test_y = test_data[:, :-1], test_data[:, -1]
    test_x_r = [x.reshape(len(x), 1) for x in test_x]
    test_data_t = [[x, y] for x, y in zip(test_x_r, test_y)]
    return test_data_t


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

############################################### 数据获取及格式处理 end ##############################

############################################ BP三层神经网络 start ################################
# 构造一个bp神经网络的类
class BP_Neural_Network:
    def __init__(self, layers):
        # 初始化
        # len_layers为层数
        self.len_layers = len(layers)
        # layers为传递进来的每层的神经元个数
        self.layers = layers
        # 初始化参数，为隐藏层和输出层每个节点都生成一个初始的b
        # 如果是3层，这里即代表两个数组，分别为隐藏层的几个神经元的b值和输出层的神经元b值
        self.b_values = [np.random.randn(n, 1) for n in layers[1:]]
        # 随机生成每条神经元连接的 weight 值
        # 如果是3层，这里即代表两个数组，分别为输入层到隐藏层的权值和隐藏层到输出层的权值
        self.w_values = [np.random.randn(x, y) for x,y in zip(layers[1:],layers[:-1])]

    # sigmoid函数
    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    # sigmoid函数的导数
    def sigmoid_derivative(self,x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)


    # 前向传播
    # 传入的a为输入值
    def forward_pass(self, a):
        # 循环传递到输出层
        for b, w in zip(self.b_values, self.w_values):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    # 反向误差传播
    # 参数x为输入的特征，y为输出值
    def back_propagation(self, x, y):
        # 初始化导数
        b_derivatives = [np.zeros(b.shape) for b in self.b_values]
        w_derivatives = [np.zeros(w.shape) for w in self.w_values]

        # 先使用前向传播算法，求得输出神经元的y值
        # a暂存x输入
        a = x
        # 存储每个神经元输出,初始为x-输入值
        a_values = [x]
        # 存储经过 sigmoid 函数计算的神经元的输入值，便于后面的直接使用和求导的使用
        z_values = []
        for b, w in zip(self.b_values, self.w_values):
            z = np.dot(w, a) + b
            z_values.append(z)
            a = self.sigmoid(z)
            a_values.append(a)

        # 先求解输出层的误差
        error = (a_values[-1]- y) * self.sigmoid_derivative(z_values[-1])
        # 根据公式进行从后往前更新
        b_derivatives[-1] = error
        w_derivatives[-1] = np.dot(error, a_values[-2].T)
        for l in range(2, self.len_layers):
            # 从后往前开始更新，因此需要采用-l
            # 利用第 l + 1 层的误差计算第 l 层的误差
            z = z_values[-l]
            zp = self.sigmoid_derivative(z)
            error = np.dot(self.w_values[-l + 1].T, error) * zp
            b_derivatives[-l] = error
            w_derivatives[-l] = np.dot(error, a_values[-l - 1].T)
        return (b_derivatives, w_derivatives)

    # 尝试使用小批量随机梯度下降策略
    def train_with_MBGD(self, train_data, count, small_batch_size, alpha, test_data,k):
        len_test = len(test_data)
        len_train = len(train_data)
        # 设定迭代次数
        for i in range(count):
            # 使训练集乱序
            random.shuffle(train_data)
            # 根据小批量样本的尺寸划分子训练集
            small_batch = [train_data[index:index + small_batch_size] for index in range(0, len_train, small_batch_size)]
            # 利用每一个小批量训练集更新 w 和 b
            for batch in small_batch:
                # 创建临时矩阵存储每次产生的临时梯度，便于求平均梯度
                temp_batch_b = [np.zeros(b.shape) for b in self.b_values]
                temp_batch_w = [np.zeros(w.shape) for w in self.w_values]
                # 求多个样本的偏导并求平均值
                for x, y in batch:
                    # 采用误差反向传播求得每一层的梯度
                    b_derivatives, w_derivatives = self.back_propagation(x, y)
                    # 累加偏导 b_derivatives, w_derivatives
                    temp_batch_b = [tb + bd for tb, bd in zip(temp_batch_b, b_derivatives)]
                    temp_batch_w = [tw + wd for tw, wd in zip(temp_batch_w, w_derivatives)]
                # 根据累加的偏导值 b_derivatives, w_derivatives 更新 b, w
                # 由于用了小样本，因此 alpha 需除以小样本长度
                self.w_values = [w - (alpha / len(batch)) * dw for w, dw in zip(self.w_values, temp_batch_w)]
                self.b_values = [b - (alpha / len(batch)) * db for b, db in zip(self.b_values, temp_batch_b)]
            # 输出相关提示信息
            print("第"+str(k)+"次验证-测试正确分类个数/总测试个数："+str(self.test(test_data))+"/"+str(len_test))
        # 返回最终测试的准确率
        return float(self.test(test_data)/len_test)


    def test(self, test_data):
        test_result = [(np.argmax(self.forward_pass(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_result)

############################################ BP三层神经网络 end ################################




# 加载训练集和测试集
train ,test=load_data()
# 创建神经网络的类并初始化
bp_NN = BP_Neural_Network([2, 4, 2])
# 进行5倍交叉验证（已提前将数据集分割并进行数据预处理，在load_data（）函数中）
acc=0.0
w_values=[]
b_values=[]
for i in range(5):
    # 获得数据，测试数据和训练数据不会重合
    # 获取本次用到的测试数据
    temp_test=test[i]
    # 获得本次用到的训练数据
    temp_tarin=train[i]
    acc+=bp_NN.train_with_MBGD(temp_tarin, 300, 5, 0.1, temp_test,i+1)
    b_values.append(bp_NN.b_values)
    w_values.append(bp_NN.w_values)

print("#########################")
print("运行结束！其中，共有80条数据，每次验证时，取其中1/5用于测试，其余4/5用于训练。每次训练后的w、b如下：")
for i in range(5):
    print("第"+str(i+1)+"次: w:"+str(w_values[i])+"  b:"+str(b_values))
print("经过5倍交叉验证，最终得到的平均准确率为：")
print(str(acc*100/5)+"%")
print("#########################")




