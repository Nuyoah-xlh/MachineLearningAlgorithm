import numpy as np
import pandas as pd
from libsvm.svmutil import *


# 数据格式转换
def dataset_format(dataset):
    ans = []
    # 将被空格分隔开的数据提取出来，并统一存储后返回
    for str in dataset:
        temp = str.split()
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        ans.append(temp)
    return ans


# 提取数据集的x特征
dataset_x = pd.read_csv("ex4x.dat", header=None)[0]
# 格式化数据
dataset_x = dataset_format(np.array(dataset_x))
# X存储libsvm格式的数据
X = []
# 将数据转换为libsvm所需的格式
for i in range(len(dataset_x)):
    temp = dataset_x[i]
    temp_dict = {}
    # 填充字典
    for j in range(len(temp)):
        temp_dict[j + 1] = temp[j]
    X.append(temp_dict)

# 提取数据集的y值
dataset_y = pd.read_csv("ex4y.dat", header=None)[0]
# 格式化数据
dataset_y = list(dataset_y)

# 参数c的取值
opt_c = [30, 50, 100, 0.1, 1, 10]
# 核函数的选择，0为线性核函数，2为径向基核函数
opt_kernel = [2, 0]
# 核函数的参数
opt_g = [0.008, 0.08, 0.5]
# 记录最高得分
best_score = -1.0
# 记录最佳参数
best_opt = ""
# 循环找出最高的准确率得分
for c_cur in opt_c:
    for kernel_cur in opt_kernel:
        for g_cur in opt_g:
            opt = "-t " + str(kernel_cur) + " -c " + str(c_cur) + " -g " + str(g_cur) + " -v 5"
            accuracy = svm_train(dataset_y, X, opt)
            if (float(accuracy) >= best_score):
                best_score = accuracy
                best_opt = opt


print("###########################################")
print("运行结束")
print("最佳参数：" + str(best_opt))
print("5折交叉验证-当前最高准确率：" + str(best_score)+"%")
print("使用最佳参数对全体数据进行训练：")
svm_train(dataset_y, X, best_opt[:19])
print("###########################################")
