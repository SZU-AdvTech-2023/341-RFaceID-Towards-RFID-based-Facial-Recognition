import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
import os
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter

tf.reset_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.45)
config.gpu_options.allow_growth = True
class_num = 100  # 分类总类数
TIME_LEN = 2  # 时间长度，每s能采样20个数据
Rows = 5
Cols = 7
data = np.load("data_RFaceID.npz", allow_pickle=True)
# 进行未进行数据增强的训练集
# 进行未进行数据增强的训练集
X_data = data["X_data"]
labels = data["labels"]
X_data = X_data.reshape((-1, 40, 70))

X_data_infulence = data["data_influence"].tolist()  ##受帽子等影响的数据
labels_influence = data["labels_influence"]

for key in X_data_infulence.keys():
    X_data_infulence[key] = X_data_infulence[key].reshape((-1, 40, 70))


def WeightBackstepAverage(inputs):
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()
    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


def filter_remove_noise(data):
    data_ = []
    for person in range(data.shape[0]):
        on_person_data = data[person].copy()
        for index in range(on_person_data.shape[1]):
            on_person_data[:, index] = WeightBackstepAverage(on_person_data[:, index])
        data_.append(on_person_data)
    return np.array(data_)


X_data = filter_remove_noise(X_data)  # data为要过滤的信号
for key in X_data_infulence.keys():
    X_data_infulence[key] = filter_remove_noise(X_data_infulence[key])


# 进行归一化
def normal(X):
    max_phase = 6.0
    max_RSSI = 60.0
    for i in range(X.shape[2]):
        if 0 == (i % 2):
            X[:, :, i] /= max_RSSI
        else:
            X[:, :, i] /= max_phase


normal(X_data)
for key in X_data_infulence.keys():
    normal(X_data_infulence[key])

plt.plot(X_data[0, :, 29])
plt.show()
plt.plot(X_data_infulence['hat'][0, :, 29])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_data, labels, test_size=0.4, random_state=666)


# 定义卷积层
def conv2d(x_, W):
    return tf.nn.conv2d(x_, W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积的输出


# 定义偏置值设置函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 标准差0.1
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 偏置增加小正值防止死亡节点
    return tf.Variable(initial)


def cnn_spatial_extractor(x_):
    with tf.name_scope("cnn"):
        ##采用CNN的方案代码
        x_image = tf.reshape(x_, [-1, Rows, Cols, 2])  # 格式：样本数*n_steps,标签的行数，标签的列数，2（RSS、phase）

        # keep_prob=tf.constant(0.5,shape=(1))
        with tf.name_scope("cnnv1"):
            # 第一个卷积层
            W_conv1 = weight_variable([3, 3, 2, 8])  # 定义一个3*3的卷积核，由2通道转化为8通道,输出为[None,Rows,Cols,8]
            b_conv1 = bias_variable([8])
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope("cnnv2"):
            # 第一个卷积层
            W_conv2 = weight_variable([3, 3, 8, 16])  # 定义一个3*3的卷积核，由8通道转化为16通道,输出为[None,Rows,Cols,16]
            b_conv2 = bias_variable([16])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        with tf.name_scope("cnnv3"):
            # 第一个卷积层
            W_conv3 = weight_variable([3, 3, 16, 32])  # 定义一个3*3的卷积核，由16通道转化为32通道,输出为[None,Rows,Cols,16]
            b_conv3 = bias_variable([32])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

        with tf.name_scope("cnnv4"):
            # 第二个卷积层
            W_conv4 = weight_variable([3, 3, 16, 8])  # 由32通道转为8通道，也可以转为其他通道数，此处只是为能好的与之前的对应
            b_conv4 = bias_variable([8])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)  # 输出为[None,Rows,Cols,8]

        with tf.name_scope("full-connect"):
            # 连一个全连接层

            W_fc1 = weight_variable([Rows * Cols * 8, 8])  # 输入为[None,Rows,Cols,8]，输出为[None,Rows,Cols,2]，None表示数据量
            #             b_fc1 = bias_variable([Rows*Cols*2])
            b_fc1 = bias_variable([8])
            h_conv4_flat = tf.reshape(h_conv4, [-1, Rows * Cols * 8])  # 把第二个卷积层的输出reshape成1Ｄ的
            h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)  # 激活函数为relu


        h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

        #         x_cnn_output=tf.reshape(h_fc1_drop,(-1,n_steps,Rows*Cols*2))   #重新形状,方便输入LSTM中
        x_cnn_output = tf.reshape(h_fc1_drop, (-1, n_steps, 8))  # 重新形状,方便输入LSTM中

        return x_cnn_output


# 使用一个双向RNN的LSTM网络
def BiRNN(x, weight, baises):
    # 将原（batch，n_step,n_input）调整为（n_step,batch,n_input）
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    #     x=tf.reshape(x,[-1,8])
    x = tf.split(x, int(n_steps))

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义正向网络
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义逆向网络

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases


batch_size = 256
display_step = 20
# 超参数
n_input = X_test.shape[2]  # 单输入形状,56个特征
n_steps = int(TIME_LEN * 20)  # 持续的步数
n_hidden = 1024  # 藏节点数
n_class = class_num  # 多分类的数目，用于定义输出的形状

tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input], name='x_input')
y = tf.compat.v1.placeholder(tf.int32, [None], name='y_input')  # 一维

# 独热码处理
y_hot = tf.one_hot(y, n_class, 1, 0)  # None*20
y_hot = tf.cast(y_hot, tf.float32)  # None*20

weights = tf.Variable(tf.compat.v1.random_normal([2 * n_hidden, n_class]))
biases = tf.Variable(tf.compat.v1.random_normal([n_class]))

# 定义输出与优化函数
# feature=cnn_spatial_extractor(x)
# pred=BiRNN(feature,weights,biases)   #输出二维
pred = BiRNN(x, weights, biases)  # 输出二维

y_prediction_softmax = tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_hot), name='loss')  # 定义损失函数

# 定义global_step
global_step = tf.Variable(0, trainable=False)

learing_rate = tf.train.exponential_decay(0.0001, global_step, 50, 0.96, staircase=False)
# learing_rate = tf.train.exponential_decay(0.001,global_step,50,0.96,staircase=False)

# 使用梯度下降算法来最优化损失值
optimizer = tf.train.AdamOptimizer(learing_rate).minimize(cost, global_step)

pred_y = tf.argmax(pred, 1, name="predict_y")  # 一维
pred_y = tf.cast(pred_y, tf.int32)
correct_pred = tf.equal(pred_y, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

init = tf.global_variables_initializer()


##获取下一个batch
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
    ### 绘制学习曲线
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
def model_run(X_train, y_train, X_test, y_test, epoches=2000):
    file_handle = open(r"F:/face code/difference_influence/time_len.txt", mode='a')
    #     file_handle.write("time_len="+str(time_len)+"s"+"\n")
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

    for key in X_data_infulence.keys():
        predict_label_influence, accuracy_influence = sess.run([pred_y, accuracy], feed_dict={x: X_data_infulence[key],
                                                                                              y: labels_influence})
        print()
        print("============================" + key + "=================================")
        print("accuracy in train dataset:", accuracy_influence)
        print('actaul label:', labels_influence)
        print('predict label:', predict_label_influence)

    sess.close()

    train_result = {"acc_train_list": acc_train_list, "acc_test_list": acc_test_list,
                    "loss_train_list": loss_train_list, "loss_test_list": loss_test_list}

    file_handle.write("acc_train_list:" + "\n")
    file_handle.write(str(acc_train_list) + "\n")
    file_handle.write("acc_test_list:" + "\n")
    file_handle.write(str(acc_test_list) + "\n")
    file_handle.write("The last accuracy is " + str(accuracy_) + "\n")
    file_handle.write("\n \n")
    file_handle.close()

    return train_result


augment_train_result = model_run(X_train, y_train, X_test, y_test, epoches=3000)

# 绘制学习曲线
plot_learning_curve(augment_train_result, "mean.png")
