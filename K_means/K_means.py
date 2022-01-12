# 数据格式转换
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 处理数据格式并返回
def dataset_format(dataset):
    ans = []
    # 将被空格分隔开的数据提取出来，并统一存储后返回
    for str in dataset:
        temp = str.split()
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        ans.append(temp)
    return ans


# 随机选择K个点作为初始的聚类中心,X_values为所有样本点，只有特征没有标签；k为簇的个数
def init(X_values, k):
    # X_num为样本数
    X_num = X_values.shape[0]
    # feature_num为每个样本的特征数
    feature_num = X_values.shape[1]
    # 定义k个聚类中心，初始为0
    cluster_centers = np.zeros((k, feature_num))
    # 每次随机取0-m的一个整数值作为下标，获取其对应的点作为聚类中心
    for i in range(k):
        # 获取0-m的一个随机整数值
        index = np.random.randint(0, X_num)
        # 获取对应的点并赋值
        cluster_centers[i, :] = X_values[index, :]
    # 返回k个聚类中心
    return cluster_centers


# 获得当前所有样本点离得最近的聚类中心的下标并返回，X_values为所有样本点；cluster_centers为所有的簇的中心点坐标
def get_nearest_cluster_centers(X_values, cluster_centers):
    # X_num为样本点数
    X_num = X_values.shape[0]
    # k为簇的数目
    k = cluster_centers.shape[0]
    # index_s用于记录每个样本离得最近的簇的中心的下标，初始为0
    index_s = np.zeros(X_num)
    # 循环遍历
    for i in range(X_num):
        # temp_min_dist用于临时记录当前样本离簇的中心的最近距离
        temp_min_dist = 2147483647
        for j in range(k):
            # 使用欧式距离进行计算和比较
            cur_dist = np.sum((X_values[i, :] - cluster_centers[j, :]) ** 2) ** 0.5
            # 如果当前距离比之前的都小，则更新
            if cur_dist < temp_min_dist:
                temp_min_dist = cur_dist
                # 记录该样本的聚类中心下标
                index_s[i] = j
    # 返回
    return index_s


# 根据每个簇的样本点更新簇的中心
def update_cluster_centers(X_values, index_s, k):
    # feature_num为每个样本的特征数
    feature_num = X_values.shape[1]
    # 初始化k个簇的中心
    cluster_centers = np.zeros((k, feature_num))
    # 分别根据每个簇的样本求每个簇的中心
    for i in range(k):
        # 找到属于该簇的所有样本点的下标并存储
        sample_of_i = np.where(index_s == i)
        # 计算平均值-新的聚类中心
        cluster_centers[i, :] = (np.sum(X_values[sample_of_i, :], axis=1) / len(sample_of_i[0]))
    # 新的簇的中心
    return cluster_centers


# 比较两次得到的簇的中心坐标是否相同
def compare(centers1, centers2):
    # k为簇的个数
    k = centers1.shape[0]
    for i in range(k):
        if centers1[i][0] != centers2[i][0]:
            return 1
        if centers1[i][1] != centers2[i][1]:
            return 1
    # 返回0说明所有数据都相等，返回1表示不相等，仍需迭代更新
    return 0


# 执行K_means算法，X_values为所有样本特征值，k为簇的个数
def K_means(X_values, k):
    # X_num为样本数
    X_num = X_values.shape[0]
    # 得到k个初始化的簇的中心
    cluster_centers = init(X_values, k)
    index_s = np.zeros(X_num)
    # 不停地迭代更新
    while True:
        # 得到当前各样本的最近的簇的中心点的下标
        index_s = get_nearest_cluster_centers(X_values, cluster_centers)
        # 如果前后两次得到的簇的中心相同，即中心不再变化，已收敛;则退出迭代
        if compare(cluster_centers, update_cluster_centers(X_values, index_s, k)) == 0:
            break
        # 更新每个簇的中心
        cluster_centers = update_cluster_centers(X_values, index_s, k)
    return (index_s, cluster_centers)


################################## 开始 #############################################

################### 提取数据并处理 ########################
# 提取数据集的x特征
dataset_x = pd.read_csv("ex4x.dat", header=None)[0]
# 格式化数据
dataset_x = dataset_format(np.array(dataset_x))
# 转换为DataFrame类型
dataset_x = pd.DataFrame(dataset_x)
# 规定其列名便于直观绘图
dataset_x.columns = ['X1', 'X2']
# 获取数据值
X_values = dataset_x.values

########绘制未聚类的散点图#######
class0 = X_values[:, :]
# 用带颜色的点区分不同的类
fig0, ax0 = plt.subplots()
ax0.scatter(class0[:, 0], class0[:, 1], label='class0')
ax0.legend()
plt.show()

################## 执行K_means算法并绘图（k=2） ########################
# 运行算法得到，index_s:样本点所属的簇；cluster_centers：簇的k个中心点
# k=2时
index_s1, cluster_centers1 = K_means(X_values, 2)
# 获取每个类对应的样本点的所有特征值用于绘图
class1 = X_values[np.where(index_s1 == 0)[0], :]
class2 = X_values[np.where(index_s1 == 1)[0], :]
# 用带颜色的点区分不同的类
fig1, ax1 = plt.subplots()
ax1.scatter(class1[:, 0], class1[:, 1], color='r', label='class-1')
ax1.scatter(class2[:, 0], class2[:, 1], color='b', label='class-2')
ax1.legend()
plt.show()

################## 执行K_means算法并绘图（k=3） ########################
# 运行算法得到，index_s:样本点所属的簇；cluster_centers：簇的k个中心点
# k=3时
index_s2, cluster_centers2 = K_means(X_values, 3)
# 获取每个类对应的样本点的所有特征值用于绘图
class1 = X_values[np.where(index_s2 == 0)[0], :]
class2 = X_values[np.where(index_s2 == 1)[0], :]
class3 = X_values[np.where(index_s2 == 2)[0], :]
# 用带颜色的点区分不同的类
fig2, ax2 = plt.subplots()
ax2.scatter(class1[:, 0], class1[:, 1], color='r', label='class-1')
ax2.scatter(class2[:, 0], class2[:, 1], color='b', label='class-2')
ax2.scatter(class3[:, 0], class3[:, 1], color='g', label='class-3')
ax2.legend()
plt.show()

################## 执行K_means算法并绘图（k=4） ########################
# 运行算法得到，index_s:样本点所属的簇；cluster_centers：簇的k个中心点
# k=4时
index_s3, cluster_centers3 = K_means(X_values, 4)
# 获取每个类对应的样本点的所有特征值用于绘图
class1 = X_values[np.where(index_s3 == 0)[0], :]
class2 = X_values[np.where(index_s3 == 1)[0], :]
class3 = X_values[np.where(index_s3 == 2)[0], :]
class4 = X_values[np.where(index_s3 == 3)[0], :]
# 用带颜色的点区分不同的类
fig3, ax3 = plt.subplots()
ax3.scatter(class1[:, 0], class1[:, 1], color='red', label='class-1')
ax3.scatter(class2[:, 0], class2[:, 1], color='blue', label='class-2')
ax3.scatter(class3[:, 0], class3[:, 1], color='green', label='class-3')
ax3.scatter(class4[:, 0], class4[:, 1], color='brown', label='class-4')
ax3.legend()
plt.show()

################## 执行K_means算法并绘图（k=5） ########################
# 运行算法得到，index_s:样本点所属的簇；cluster_centers：簇的k个中心点
# k=5时
index_s4, cluster_centers4 = K_means(X_values, 5)
# 获取每个类对应的样本点的所有特征值用于绘图
class1 = X_values[np.where(index_s4 == 0)[0], :]
class2 = X_values[np.where(index_s4 == 1)[0], :]
class3 = X_values[np.where(index_s4 == 2)[0], :]
class4 = X_values[np.where(index_s4 == 3)[0], :]
class5 = X_values[np.where(index_s4 == 4)[0], :]
# 用带颜色的点区分不同的类
fig4, ax4 = plt.subplots()
ax4.scatter(class1[:, 0], class1[:, 1], color='red', label='class-1')
ax4.scatter(class2[:, 0], class2[:, 1], color='blue', label='class-2')
ax4.scatter(class3[:, 0], class3[:, 1], color='green', label='class-3')
ax4.scatter(class4[:, 0], class4[:, 1], color='brown', label='class-4')
ax4.scatter(class5[:, 0], class5[:, 1], color='black', label='class-5')
ax4.legend()
plt.show()
print("####################################")
print("运行结束，结果如下")
print("k=2时，得到2个聚类中心：\n" + cluster_centers1.__str__())
print("k=3时，得到3个聚类中心：\n" + cluster_centers2.__str__())
print("k=4时，得到4个聚类中心：\n" + cluster_centers3.__str__())
print("k=5时，得到5个聚类中心：\n" + cluster_centers4.__str__())
print("可视化结果见plt绘图")
print("注：如程序异常结束，则为随机产生的聚类中心点不合适导致运算异常，重新运行即可解决问题！")
