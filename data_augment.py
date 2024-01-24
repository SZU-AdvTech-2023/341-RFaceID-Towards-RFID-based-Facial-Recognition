# Data Augmentation
# 1. 将原始数据切分为训练集和测试集
# 2. 对训练集和测试集进行数据增强，形成的合成数据集和原来的数据集组成新的增强数据集
# 3. 保存数据

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.utils import to_time_series_dataset
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesResampler

CLASS_NAME = 20

# 1. 将原始数据切分为训练集和测试集

# 加载数据
# 从original_data_RFaceID.npz文件中加载的数据

data = np.load("original_data_RFaceID.npz")
# 未进行数据增强的训练集
X_data = data["X_data"]
labels = data["labels"]
# 将数据进行切分，每40个采样点作为一个数据  其中偶数下标为RSS 奇数下标为phase
X_data = X_data.reshape((-1, 40, 70))

plt.rcParams['figure.figsize'] = (2.0, 80.0)
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 设置水平间隔和垂直间隔
for i in range(1, X_data.shape[2], 2):  # 35个图
    plt.subplot(35, 1, int(i / 2 + 1))
    plt.plot(X_data[0, :, i])  # 分别取第1维第一个、第2维全部(一个窗口)、第3维第i（1,3,...,69）个数据
    plt.title(str(i))
plt.show()

# 数据划分
# 将数据按指定比例划分为训练集和测试集
# 现在把X_data和对应的labels随机分为各5000条，作为训练集和测试集
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_data, labels, test_size=0.3,
                                                                                        random_state=666)


# 2. 对训练集和测试集进行数据增强，形成的合成数据集和原来的数据集组成新的增强数据集
# 产生一个尺寸为size的随机数组，用以表示每个对象在生成合成对象中所占的权重，故数组的所有元素和为1.0
def generate_random_wight(size=3):  # [0.54426761 0.00247134 0.45326105]
    random_wight = np.random.uniform(size=size)  # 均匀分布
    sum_ = np.sum(random_wight)
    random_wight = random_wight / sum_
    return random_wight


# 从一个类中产生一个合成数据
def generate_synthetic_one_data(one_label_data, label, size=3):
    if one_label_data.ndim != 3:  # 维数不为3则num_size设为1
        num_size = 1
    else:  # 维数为3则num_size设为one_label_data的长度
        num_size = len(one_label_data)
        # 断言 不满足条件则输出错误信息
    assert num_size >= size, label + "类数据量不够异常:size=" + size + ",但是用于合成的数据数量只有n=" + num_size

    index = np.random.choice(a=num_size, size=size, replace=False)  # 不重复抽样,随机从40个里抽取
    original_data = one_label_data[index]  # 选出用于合成数据的原始数据
    formatted_dataset = to_time_series_dataset(original_data)  # 格式化数据，符合tslearn数据格式

    random_weight = generate_random_wight(size)
    # random_weight = np.array([0.8,0.1,0.1])
    # DTW 从原数据中合成新数据
    syn_data = dtw_barycenter_averaging(formatted_dataset, max_iter=50, barycenter_size=original_data.shape[1],
                                        weights=random_weight)
    return syn_data


# 产生所有合成数据
def generate_synthetic_all_data(X_data, y_labels, size=3, percentage=0.3):
    """
        X_data:用于合成数据的原始数据集
        y_labels:用于合成数据的原始标签集
        size:用size个原始数据来合成一个合成数据
        percentage:合成数据的总数占原始数据数量的百分比
    """
    syn_labels = []
    syn_data = []
    labels_num = CLASS_NAME  # 共计100个类
    nums = int(len(y_labels) / labels_num * percentage)  # 表示每一个类需要合成数据的数量
    for label in range(CLASS_NAME):  # 表示从[0,100)共计100个类，每个类都要合成数据
        for num in range(nums):
            temp_syn_data = generate_synthetic_one_data(X_data[y_labels == label], label,
                                                        size=size)  # 由一类的多个数据产生一个综合的数据
            syn_labels.append(label)
            syn_data.append(temp_syn_data)  # 将合成数据添加到数组中

    return np.array(syn_data), np.array(syn_labels)


syn_X_train_data, syn_y_train_data = generate_synthetic_all_data(X_train_original, y_train_original, size=3)
syn_X_test_data, syn_y_test_data = generate_synthetic_all_data(X_test_original, y_test_original, size=3)

# 将合成的数据集加入测试集
X_test_syn = np.vstack((X_test_original, syn_X_test_data))
y_test_syn = np.array(list(y_test_original) + list(syn_y_test_data))

# 将合成的数据集加入训练集
X_train_syn = np.vstack((X_train_original, syn_X_train_data))
y_train_syn = np.array(list(y_train_original) + list(syn_y_train_data))

# 绘制合成的训练集数据与原始的数据的做对比
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
subfig1.set_xlabel('Sample Index')
i = 1
j = 0
subfig1.plot(X_data[j, :, i], 'r', label='original'), print(labels[1])
subfig1.plot(syn_X_train_data[j, :, i], 'g', label='synthesis'), print(syn_y_train_data[1])
subfig1.legend()
i = 0  # 46
subfig2.set_xlabel('Sample Index')
subfig2.plot(X_data[j, :, i], 'r', label='original'), print(labels[1])
subfig2.plot(syn_X_train_data[j, :, i], 'g', label='synthesis'), print(syn_y_train_data[1])
subfig2.legend()
plt.show()


# ②添加噪音
# 解决不同环境下环境噪音（多径反射）问题

# 向数据中添加均值为mu，方差为sigma的高斯噪音
def add_noise(X_data, phase_sigma=0.05, RSSI_sigma=0.5, mu=0.0):
    noise = np.zeros(X_data.shape)
    size = noise[:, :, 0].shape
    for i in range(X_data.shape[2]):
        if i % 2 == 0:
            noise[:, :, i] = np.random.normal(mu, RSSI_sigma, size)
        else:
            noise[:, :, i] = np.random.normal(mu, phase_sigma, size)

    return noise + X_data


# 给训练集和测试集添加噪声
X_train_syn_noise = add_noise(X_train_syn)
X_test_syn_noise = add_noise(X_test_syn)

# 绘制添加噪声和没添加噪声的数据
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
# phase
j = 0
i = 1
subfig1.set_xlabel('Sample Index')
# plt.plot(X_train_original[j,:,i],'r')
subfig1.plot(X_train_syn[j, :, i], 'r', label='before')
subfig1.plot(X_train_syn_noise[j, :, i], 'g', label='after')
subfig1.legend()
# RSS
i = 0
subfig2.set_xlabel('Sample Index')
# plt.plot(X_train_original[j,:,i],'r')
subfig2.plot(X_train_syn[j, :, i], 'r', label='before')
subfig2.plot(X_train_syn_noise[j, :, i], 'g', label='after')
subfig2.legend()
fig.savefig('noise.png')
plt.show()


# ③距离变化缩放
# 解决同一个人与标签的距离不同的问题
# 超参数：这里增大的百分比γ的取值范围为（-15%~+15%）
def scaling(X_data, uplimit=0.15, downlimit=-0.15):
    scale_data = []
    for data in X_data:
        scale = (np.random.random() * (uplimit - downlimit) + downlimit) + 1
        scale_data.append(scale * data)

    return np.array(scale_data)


# 在添加噪音的基础上改变数据大小
X_train_syn_noise_scale = scaling(X_train_syn_noise)
X_test_syn_noise_scale = scaling(X_test_syn_noise)

# 绘制改变数据大小后的数据与仅添加噪声的数据做对比
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
j = 0
i = 1
subfig1.set_xlabel('Sample Index')
subfig1.plot(X_train_syn_noise[j, :, i], 'r', label='before')
subfig1.plot(X_train_syn_noise_scale[j, :, i], 'g', label='after')
subfig1.legend()
i = 0
subfig2.set_xlabel('Sample Index')
subfig2.plot(X_train_syn_noise[j, :, i], 'r', label='before')
subfig2.plot(X_train_syn_noise_scale[j, :, i], 'g', label='after')
subfig2.legend()
fig.savefig('scale.png')
plt.show()


# ④数据拉伸/压缩
# 解决晃头时速度不同的问题

# 这里的压缩选取（-30% - +30%）的随机形变
# 形变后还要进行数据对齐，对齐的方案按之前的解决
def deformation_series(X_data, uplimit=0.5, downlimit=-0.5):
    deformation_data = []
    for series in X_data:
        size = int((np.random.random() * (uplimit - downlimit) + downlimit) * X_data.shape[1]) + X_data.shape[1]
        series = series[np.newaxis, :, :]
        stretch_series = TimeSeriesResampler(sz=size).fit_transform(series)  # 重采样，拉伸/压缩
        stretch_series = stretch_series[0]

        # 对齐
        if size > X_data.shape[1]:
            stretch_series = stretch_series[:X_data.shape[1]]
        elif size < X_data.shape[1]:
            stretch_series = np.vstack((stretch_series, stretch_series[:(X_data.shape[1] - size), :]))

        deformation_data.append(stretch_series)

    return np.array(deformation_data)


# 在添加噪声和改变数据大小的基础上进行拉伸压缩
X_train_syn_noise_scale_deformation = deformation_series(X_train_syn_noise_scale)
X_test_syn_noise_scale_deformation = deformation_series(X_test_syn_noise_scale)

# 绘制拉伸压缩后的数据与前面的数据做对比
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
j = 0
subfig1.set_xlabel('Sample Index')
i = 1
subfig1.plot(X_train_syn_noise_scale[j, :, i], 'r', label='before')
subfig1.plot(X_train_syn_noise_scale_deformation[j, :, i], 'g', label='after')
subfig1.legend()

subfig2.set_xlabel('Sample Index')
i = 0
subfig2.plot(X_train_syn_noise_scale[j, :, i], 'r', label='before')
subfig2.plot(X_train_syn_noise_scale_deformation[j, :, i], 'g', label='after')
subfig2.legend()
fig.savefig('deformation.png')
plt.show()


# ⑤翻转数据
# 解决一个人由于左右晃头的方向不同，而造成的差异

# 超参数η：对多少（百分比）的数据进行翻转  这里设定为20%

def reversal_data(X_data, percentage=0.2):
    size_ = int(len(X_data) * percentage)  # 获得需要翻转的数据量
    index = np.random.choice(a=len(X_data), size=size_, replace=False)  # 不重复抽样,随机抽取size个数据的索引
    copy_data = X_data.copy()  # 创建副本,防止对原数据的改变
    copy_data[index] = copy_data[index, ::-1, :]  # 对选定的索引样本进行翻转
    return copy_data


# 在拉伸压缩的基础上进行数据翻转
X_train_syn_noise_scale_deformation_rev = reversal_data(X_train_syn_noise_scale_deformation)
X_test_syn_noise_scale_deformation_rev = reversal_data(X_test_syn_noise_scale_deformation)

# 绘制翻转后的数据与前面的数据做对比
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
j = 0
subfig1.set_xlabel('Sample Index')
i = 1
subfig1.plot(X_train_syn_noise_scale_deformation[j, :, i], 'r', label='before')
subfig1.plot(X_train_syn_noise_scale_deformation_rev[j, :, i], 'g', label='after')
subfig1.legend()

subfig2.set_xlabel('Sample Index')
i = 0
subfig2.plot(X_train_syn_noise_scale_deformation[j, :, i], 'r', label='before')
subfig2.plot(X_train_syn_noise_scale_deformation_rev[j, :, i], 'g', label='after')
subfig2.legend()
fig.savefig('reversal.png')
plt.show()

# 综合实现上述数据增强
# X_data
X_train_augment = X_train_syn.copy()
X_train_augment = add_noise(X_train_augment)
X_train_augment = scaling(X_train_augment)
X_train_augment = deformation_series(X_train_augment)
X_train_augment = reversal_data(X_train_augment)

X_test_augment = X_test_syn.copy()
X_test_augment = add_noise(X_test_augment)
X_test_augment = scaling(X_test_augment)
X_test_augment = deformation_series(X_test_augment)
X_test_augment = reversal_data(X_test_augment)
# 将增强的数据添加进数据集，并写入文件

# 将数据以字典的形式压缩至augment_data.npz文件中
np.savez("augment_data.npz", X_train_augment=X_train_augment, X_test_augment=X_test_augment,
         X_train_no_augment=X_train_original,
         X_test_no_augment=X_test_original, y_train_augment=y_train_syn, y_test_augment=y_test_syn,
         y_train_no_augment=y_train_original, y_test_no_augment=y_test_original)
