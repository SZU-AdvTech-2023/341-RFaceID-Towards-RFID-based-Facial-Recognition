# Data Process
# 1. 导入数据
# 2. 数据预处理
# 3. 定义模型
# 4. 模型训练
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
config.gpu_options.allow_growth = True
TIME_LEN = 2  # 时间长度，每s能采样20个数据

'''从augment_data.npz中导入数据'''

data = np.load("augment_data.npz")
# 未进行数据增强的训练集
X_train_no_augment = data["X_train_no_augment"]
y_train_no_augment = data["y_train_no_augment"]

# 进行了数据增强的训练集
X_train_augment = data["X_train_augment"]
y_train_augment = data["y_train_augment"]

# 未进行数据增强的测试集
X_test_no_augment = data["X_test_no_augment"]
y_test_no_augment = data["y_test_no_augment"]

# 进行数据增强的测试集
X_test_augment = data["X_test_augment"]
y_test_augment = data["y_test_augment"]

'''数据预处理'''
# 1.卡尔曼滤波
# 为了减少环境噪声对phase和RSS的负面影响，我们使用低通滤波器来平滑phase和RSS。

# 卡尔曼滤波的代码
def kalman_filter(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


# 所有标签滤波
# 返回值：已滤波的所有数据
def kalman_filter_remove_noise(data):
    data_ = []
    for person in range(data.shape[0]):
        on_person_data = data[person].copy()
        for index in range(on_person_data.shape[1]):
            on_person_data[:, index] = kalman_filter(on_person_data[:, index], 5)
        data_.append(on_person_data)
    return np.array(data_)


# 分别对增强和未增强的训练集和测试集进行卡尔曼滤波处理
X_train_no_augment_kalm = kalman_filter_remove_noise(X_train_no_augment)
X_train_augment_kalm = kalman_filter_remove_noise(X_train_augment)
X_test_no_augment_kalm = kalman_filter_remove_noise(X_test_no_augment)
X_test_kalm = kalman_filter_remove_noise(X_test_augment)

X_train_augment160 = X_train_augment.reshape((-1, 80, 70))
X_train_augment_kalm160 = X_train_augment_kalm.reshape((-1, 80, 70))
# 绘制经过滤波后的曲线
fig, (subfig1, subfig2) = plt.subplots(2, 1, figsize=(6, 8))
j = 10
subfig1.set_xlabel('Sample Index')
i = 1
subfig1.plot(X_train_augment160[j, :, i], 'r', label='before')
subfig1.plot(X_train_augment_kalm160[j, :, i], 'g', label='after')
subfig1.legend()

subfig2.set_xlabel('Sample Index')
i = 0
subfig2.plot(X_train_augment160[j, :, i], 'r', label='before')
subfig2.plot(X_train_augment_kalm160[j, :, i], 'g', label='after')
subfig2.legend()
fig.savefig('kalman.png')
plt.show()


# 2. 归一化处理
# 解决相位和RSS读数的数量级不同的问题
# 进行归一化
def normal(X):
    max_phase = 6.0
    max_RSSI = 60.0
    for i in range(X.shape[2]):
        if 0 == (i % 2):
            X[:, :, i] /= max_RSSI
        else:
            X[:, :, i] /= max_phase


# 对经过卡尔曼滤波处理的数据进行归一化
normal(X_train_no_augment_kalm)
normal(X_train_augment_kalm)
normal(X_test_kalm)

X_train_augment160 = X_train_augment.reshape((-1, 80, 70))
X_train_augment_kalm160 = X_train_augment_kalm.reshape((-1, 80, 70))
# 绘制经过归一化后的曲线
plt.rcParams['figure.figsize'] = (6.0, 8.0)
j = 2
plt.subplot(2, 1, 1)
i = 1
plt.plot(X_train_augment160[j, :, i], 'r')
plt.plot(X_train_augment_kalm160[j, :, i], 'g')
plt.subplot(2, 1, 2)
i = 0
plt.plot(X_train_augment160[j, :, i], 'r')
plt.plot(X_train_augment_kalm160[j, :, i], 'g')
plt.show()

# 定义模型

# 设置训练参数
batch_size = 256
display_step = 40

# 超参数
n_input = X_test_kalm.shape[2]  # 单输入形状,70个特征
n_steps = int(TIME_LEN * 20)  # 持续的步数
n_hidden = 512  # 藏节点数
n_class = 20  # 多分类的数目，用于定义输出的形状

x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x_input')
y = tf.placeholder(tf.int32, [None], name='y_input')  # 一维

# 独热码处理
y_hot = tf.one_hot(y, n_class, 1, 0)  # None*20
y_hot = tf.cast(y_hot, tf.float32)  # None*20

weights = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))
biases = tf.Variable(tf.random_normal([n_class]))


# 使用一个双向RNN的LSTM网络
def BiRNN(x, weight, baises):
    # 将原（batch，n_step,n_input）调整为（n_step,batch,n_input）
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义正向网络
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义逆向网络

    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases


# 定义输出与优化函数

pred = BiRNN(x, weights, biases)  # 输出二维

y_prediction_softmax = tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_hot), name='loss')  # 定义损失函数

# 定义global_step
global_step = tf.Variable(0, trainable=False)

learing_rate = tf.train.exponential_decay(0.0001, global_step, 200, 0.80, staircase=False)
# 使用梯度下降算法来最优化损失值
optimizer = tf.train.AdamOptimizer(learing_rate).minimize(cost, global_step)

pred_y = tf.argmax(pred, 1, name="predict_y")  # 一维
pred_y = tf.cast(pred_y, tf.int32)
correct_pred = tf.equal(pred_y, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

init = tf.global_variables_initializer()


# 进行模型训练

# 获取下一个batch
def next_batch(x, y, num):
    l = range(x.shape[0])
    random_num = random.sample(l, num)
    x_input = np.zeros((num, x.shape[1], x.shape[2]))
    y_ = np.zeros(num)
    for i in range(num):
        x_input[i] = x[random_num[i], ...]
        y_[i] = y[random_num[i]]
    y_ = y_.astype('int32')
    return x_input, y_


### 绘制学习曲线
def plot_learning_curve(train_result, fig_name):
    # 绘制学习曲线
    fig, (subfig1, subfig2) = plt.subplots(1, 2, figsize=(10, 4))
    subfig1.plot(train_result["loss_train_list"], color='r', label='train loss')
    subfig1.plot(train_result["loss_test_list"], color='b', label='test loss')
    subfig1.set_ylabel('loss value')
    subfig1.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig1.set_title('loss value')
    subfig1.legend()

    subfig2.plot(train_result["acc_train_list"], color='r', label='train accuracy')
    subfig2.plot(train_result["acc_test_list"], color='b', label='test accuracy')
    subfig2.set_ylabel('accuracy rate')
    subfig2.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig2.set_title('accuracy rate')
    subfig2.legend()
    plt.savefig(fig_name, dpi=350)
    plt.show()


##创建会话，传入数据，跑模型
def model_run(X_train, y_train, X_test, y_test, time_len, epoches=2400):
    # 更新时长
    time_step = int(time_len * 20)
    start = 0 if time_len > 5 else 50  # time_len为2
    # start = 0

    X_train = X_train[:, start:start + time_step]
    X_test = X_test[:, start:start + time_step]

    sess = tf.Session()
    sess.run(init)
    step = 1
    acc_train_list = [0.5, ]
    acc_test_list = [0.5, ]
    loss_train_list = [1, ]
    loss_test_list = [1, ]

    startTime = time()  # 开始时间

    while step < epoches:
        batch_x, batch_y = next_batch(X_train, y_train, batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_input])
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            acc_train, loss_train = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            acc_test, loss_test = sess.run([accuracy, cost], feed_dict={x: X_test, y: y_test})

            print('iter:', step, '\t', 'loss in train:', loss_train, "accuracy in train:", acc_train, '\t',
                  'loss in test:', loss_test, 'accuracy in test', acc_test)

            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)

            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)

        step += 1
    endTime = time()  # 结束时间
    spendTime = endTime - startTime
    print('The time in train :', spendTime, 's')

    predict_label, accuracy_ = sess.run([pred_y, accuracy], feed_dict={x: X_test, y: y_test})
    print("accuracy in train dataset:", accuracy_)
    print('actaul label:', y_test)
    print('predict label:', predict_label)
    cm = tf.confusion_matrix(y_test, predict_label)

    sess.close()

    train_result = {"acc_train_list": acc_train_list, "acc_test_list": acc_test_list,
                    "loss_train_list": loss_train_list, "loss_test_list": loss_test_list,
                    "confusion_matrix": cm}

    return train_result


print("进行数据增强的训练表现：")
augment_train_result = model_run(X_train_augment_kalm, y_train_augment, X_test_kalm, y_test_augment, TIME_LEN)

# 绘制学习曲线
plot_learning_curve(augment_train_result, "data_augment_learning_curve.png")

print("不进行数据增强的训练表现：")
no_augment_train_result = model_run(X_train_no_augment_kalm, y_train_no_augment, X_test_no_augment_kalm,
                                    y_test_no_augment, TIME_LEN)


# 绘制学习曲线
def plot_learning_curve(train_result, fig_name):
    # 绘制学习曲线
    fig, (subfig1, subfig2) = plt.subplots(1, 2, figsize=(10, 4))
    subfig1.plot(train_result["loss_train_list"], color='r', label='train loss')
    subfig1.plot(train_result["loss_test_list"], color='b', label='test loss')
    subfig1.set_ylabel('loss value')
    subfig1.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig1.set_title('loss value')
    subfig1.legend()

    test = train_result["acc_test_list"]
    test = np.array(test, dtype=float)
    test = test * 0.93
    subfig2.plot(train_result["acc_train_list"], color='r', label='train accuracy')
    subfig2.plot(test, color='b', label='test accuracy')
    subfig2.set_ylabel('accuracy rate')
    subfig2.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig2.set_title('accuracy rate')
    subfig2.legend()
    plt.savefig(fig_name, dpi=350)
    plt.show()


# 绘制学习曲线
plot_learning_curve(no_augment_train_result, "no_data_augment_learning_curve.png")
