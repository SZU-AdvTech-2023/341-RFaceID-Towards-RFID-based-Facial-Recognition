import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
import sklearn as sk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TIME_LEN = 2  # 时间长度，每s能采样20个数据
time_step = TIME_LEN * 20

# 导入数据
data = np.load("./no_augment_data.npz", allow_pickle=True)
# 进行未进行数据增强的训练集
X_train_augment = data["X_train"]
y_train_augment = data["y_train"]
X_test_augment = data['X_test']
y_test_augment = data['y_test']

data = np.load("./augment_data.npz")
# 进行未进行数据增强的训练集
X_train_no_augment = data["X_train_no_augment"]
y_train_no_augment = data["y_train_no_augment"]

# 测试集
X_test_no_augment = data["X_test_augment"]
y_test_no_augment = data["y_test_augment"]

# 采用SVM解决
svc_clf = SVC(kernel='rbf', gamma=0.005)
svc_clf.fit(X_train_augment.reshape(X_train_augment.shape[0], -1), y_train_augment)
svc_clf.score(X_test_augment.reshape(X_test_augment.shape[0], -1), y_test_augment)

# 采用高斯核函数
gamma_list = np.arange(1, 12, 2) * 0.01
scores = []
print(gamma_list)
for i in gamma_list:
    svc_clf = SVC(kernel="rbf", gamma=i)
    svc_clf.fit(X_train_augment.reshape(X_train_augment.shape[0], -1), y_train_augment)
    score = svc_clf.score(X_test_augment.reshape(X_test_augment.shape[0], -1), y_test_augment)
    print(score)
    scores.append(score)
plt.plot(gamma_list, scores)
plt.show()

svc_clf = SVC(kernel='rbf', gamma=0.005)
svc_clf.fit(X_train_no_augment.reshape(X_train_no_augment.shape[0], -1), y_train_no_augment)
svc_clf.score(X_test_no_augment.reshape(X_test_no_augment.shape[0], -1), y_test_no_augment)

# 采用高斯核函数
gamma_list = np.arange(1, 12, 2) * 0.01
scores = []
print(gamma_list)
for i in gamma_list:
    svc_clf = SVC(kernel="rbf", gamma=i)
    svc_clf.fit(X_train_no_augment.reshape(X_train_no_augment.shape[0], -1), y_train_no_augment)
    score = svc_clf.score(X_test_no_augment.reshape(X_test_no_augment.shape[0], -1), y_test_no_augment)
    print(score)
    scores.append(score)
plt.plot(gamma_list, scores)
plt.show()

# 随机森林（RF）
# 采用高斯核函数
depth_list = np.arange(5, 50, 5)
scores = []
print(depth_list)
for i in depth_list:
    print(i)
    rf_clf = RandomForestClassifier(max_depth=i, n_estimators=1000, oob_score=False)
    rf_clf.fit(X_train_augment.reshape(X_train_augment.shape[0], -1), y_train_augment)
    socre = rf_clf.score(X_test_augment.reshape(X_test_augment.shape[0], -1), y_test_augment)
    print(score)
    scores.append(score)
plt.plot(depth_list, scores)
plt.show()
rf_clf = RandomForestClassifier(max_depth=15, n_estimators=400, random_state=666, oob_score=False)
rf_clf.fit(X_train_no_augment.reshape(X_train_no_augment.shape[0], -1), y_train_no_augment)
rf_clf.score(X_test_no_augment.reshape(X_test_no_augment.shape[0], -1), y_test_no_augment)
# 采用高斯核函数
depth_list = np.arange(5, 50, 5)
scores = []
print(depth_list)
for i in depth_list:
    rf_clf = RandomForestClassifier(max_depth=i, n_estimators=400, random_state=666, oob_score=False)
    rf_clf.fit(X_train_no_augment.reshape(X_train_no_augment.shape[0], -1), y_train_no_augment)
    score = rf_clf.score(X_test_no_augment.reshape(X_test_no_augment.shape[0], -1), y_test_no_augment)
    print(score)
    scores.append(score)
plt.plot(depth_list, scores)
plt.show()

# 逻辑回归
lr = LogisticRegression(C=20.0, random_state=1, n_jobs=-1, solver='sag', max_iter=200)
lr.fit(X_train_augment.reshape(X_train_augment.shape[0], -1), y_train_augment)
lr.score(X_test_augment.reshape(X_test_augment.shape[0], -1), y_test_augment)
# 采用高斯核函数
C_list = np.arange(1, 105, 20)
scores = []
print(C_list)
for i in C_list:
    lr = LogisticRegression(C=i, random_state=1, n_jobs=-1, solver='sag')
    lr.fit(X_train_augment.reshape(X_train_augment.shape[0], -1), y_train_augment)
    score = lr.score(X_test_augment.reshape(X_test_augment.shape[0], -1), y_test_augment)
    print(score)
    scores.append(score)
plt.plot(C_list, scores)
plt.show()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_no_augment.reshape(X_train_no_augment.shape[0], -1), y_train_no_augment)
lr.score(X_test_no_augment.reshape(X_test_no_augment.shape[0], -1), y_test_no_augment)

#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CLASS_NAME = 100  # 分类总类数
TIME_LEN = 2  # 时间长度，每s能采样20个数据
# 设置训练参数
batch_size = 256
display_step = 20

Rows = 5  # 标签矩阵行数
Cols = 7  # 标签矩阵列数

# 超参数
n_input = X_train_augment.shape[2]  # 单输入形状,70个特征
n_steps = int(TIME_LEN * 20)  # 持续的步数
n_hidden = 1024  # 藏节点数
n_class = int(CLASS_NAME)  # 多分类的数目，用于定义输出的形状


# 定义卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积的输出


# 定义偏置值设置函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 标准差0.1
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 偏置增加小正值防止死亡节点
    return tf.Variable(initial)


def cnn_spatial_extractor(x):
    with tf.name_scope("cnn"):
        #采用CNN的方案代码
        x_image = tf.reshape(x, [-1, Rows, Cols, 2])  # 格式：样本数*n_steps,标签的行数，标签的列数，2（RSS、phase）

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
            W_conv4 = weight_variable([3, 3, 32, 8])  # 由32通道转为8通道，也可以转为其他通道数，此处只是为能好的与之前的对应
            b_conv4 = bias_variable([8])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)  # 输出为[None,Rows,Cols,8]
            print("h_conv4:" + str(h_conv4))

        with tf.name_scope("full-connect1"):
            # 连一个全连接层
            W_fc1 = weight_variable(
                [Rows * Cols * 8 * 40, 128])  # 输入为[None,Rows,Cols,8]，输出为[None,Rows,Cols,2]，None表示数据量
            b_fc1 = bias_variable([128])
            h_conv4_flat = tf.reshape(h_conv4, [-1, 40 * Rows * Cols * 8])  # 把第二个卷积层的输出reshape成1Ｄ的
            h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)  # 激活函数为relu
            h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)  # shape=(None,128)
            print("h_conv4:" + str(h_fc1_drop))

        with tf.name_scope("full-connect2"):
            # 连一个全连接层
            W_fc2 = weight_variable([128, n_class])  # 输入为[None,Rows,Cols,8]，输出为[None,Rows,Cols,2]，None表示数据量
            b_fc2 = bias_variable([n_class])
            h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # 激活函数为relu
            # h_fc2_drop = tf.nn.dropout(h_fc2,0.5)

        return h_fc2


# 使用一个双向RNN的LSTM网络
def BiRNN_temporal_extractor(x):
    with tf.name_scope("BI-LSTM"):
        weights = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))
        biases = tf.Variable(tf.random_normal([n_class]))
        # 将原（batch，n_step,n_input）调整为（n_step,batch,n_input）
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_input])
        x = tf.split(x, n_steps)

        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义正向网络
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)  # 定义逆向网络

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights) + biases  # shape=(None(batch-size),n_class)


# 输入
with tf.name_scope(name='input'):
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x_input')
    y = tf.placeholder(tf.int32, [None], name='y_input')  # 一维#独热码处理
    y_hot = tf.one_hot(y, n_class, 1, 0)  # None*20
    y_hot = tf.cast(y_hot, tf.float32)  # None*20

# 隐藏层
with tf.name_scope(name='CNN_hidden'):
    predictor = cnn_spatial_extractor(x)

# 隐藏层
with tf.name_scope(name='BI-LSTM_hidden'):
    predictor_by_LSTM = BiRNN_temporal_extractor(x)

# 输出层
with tf.name_scope(name='output'):
    y_prediction_softmax = tf.nn.softmax(predictor)
    pred_y = tf.argmax(predictor, 1, name="predict_y_LSTM")  # 一维
    pred_y = tf.cast(pred_y, tf.int32)
    correct_pred = tf.equal(pred_y, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_CNN')

    y_prediction_softmax_LSTM = tf.nn.softmax(predictor_by_LSTM)
    pred_y_LSTM = tf.argmax(predictor_by_LSTM, 1, name="predict_y_LSTM")  # 一维
    pred_y_LSTM = tf.cast(pred_y_LSTM, tf.int32)
    correct_pred = tf.equal(pred_y_LSTM, y)
    accuracy_LSTM = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_LSTM')

# 优化器
pred_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictor, labels=y_hot), name='pred_loss')
pred_loss_LSTM = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictor_by_LSTM, labels=y_hot),
                                name='pred_loss')

# 定义global_step，指数下降的学习率
global_step_cls = tf.Variable(0, trainable=False)
learing_rate_cls = tf.train.exponential_decay(0.0012, global_step_cls, 20, 0.96, staircase=False)

global_step_LSTM = tf.Variable(0, trainable=False)
learing_rate_LSTM = tf.train.exponential_decay(0.00011, global_step_LSTM, 50, 0.96, staircase=False)

optimizer = tf.train.AdamOptimizer(learing_rate_cls, name="optimizer_CNN").minimize(pred_loss, global_step_cls)
optimizer_LSTM = tf.train.AdamOptimizer(learing_rate_LSTM, name="optimizer_LSTM").minimize(pred_loss_LSTM,
                                                                                           global_step_LSTM)

# 初始化整个网络
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
    subfig1.plot(train_result["loss_train_list"][1:], color='r', label='train loss')
    subfig1.plot(train_result["loss_test_list"][1:], color='b', label='test loss')
    subfig1.set_ylabel('loss value')
    subfig1.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig1.set_title('loss value')
    subfig1.legend()

    subfig2.plot(train_result["acc_train_list"][1:], color='r', label='train accuracy')
    subfig2.plot(train_result["acc_test_list"][1:], color='b', label='test accuracy')
    subfig2.set_ylabel('accuracy rate')
    subfig2.set_xlabel('iter num(x' + str(display_step) + ')')
    subfig2.set_title('accuracy rate')
    subfig2.legend()
    # plt.savefig(fig_name,dpi=350)
    plt.show()


##创建会话，传入数据，跑模型
def model_run(X_train, y_train, X_test, y_test, topic="CNN", epoches=2000):
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
        # print(batch_x.shape)
        if (topic == "CNN"):
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        else:
            sess.run(optimizer_LSTM, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            if (topic == "CNN"):
                acc_train, loss_train = sess.run([accuracy, pred_loss], feed_dict={x: batch_x, y: batch_y})
                acc_test, loss_test = sess.run([accuracy, pred_loss], feed_dict={x: X_test, y: y_test})
            else:
                acc_train, loss_train = sess.run([accuracy_LSTM, pred_loss_LSTM], feed_dict={x: batch_x, y: batch_y})
                acc_test, loss_test = sess.run([accuracy_LSTM, pred_loss_LSTM], feed_dict={x: X_test, y: y_test})
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

    predict_label, accuracy_ = sess.run([pred_y_LSTM, accuracy_LSTM], feed_dict={x: X_test, y: y_test})
    print("accuracy in train dataset:", accuracy_)
    print('actaul label:', y_test)
    print('predict label:', predict_label)
    cm = tf.confusion_matrix(y_test, predict_label)

    sess.close()

    train_result = {"acc_train_list": acc_train_list, "acc_test_list": acc_test_list,
                    "loss_train_list": loss_train_list, "loss_test_list": loss_test_list,
                    "confusion_matrix": cm}

    return train_result


# 仅CNN
print("进行数据增强的训练表现：")
augment_train_result = model_run(X_train_augment, y_train_augment, X_test_augment, y_test_augment)
# 绘制学习曲线
plot_learning_curve(augment_train_result, "data_augment_learning_curve.png")
print("不进行进行数据增强的训练表现：")
no_augment_train_result = model_run(X_train_augment, y_train_augment, X_test_augment, y_test_augment, topic="LSTM",
                                    epoches=4000)
# 绘制学习曲线
plot_learning_curve(no_augment_train_result, "no_data_augment_learning_curve.png")

# 仅BI—LSTM
print("进行数据增强的训练表现：")
augment_train_result = model_run(X_train_no_augment, y_train_no_augment, X_test_no_augment, y_test_no_augment,
                                 topic="LSTM", epoches=4000)
# 绘制学习曲线
plot_learning_curve(augment_train_result, "data_augment_learning_curve.png")
no_augment = no_augment_train_result.copy()
# 绘制学习曲线
plot_learning_curve(no_augment_train_result, "data_no_augment_learning_curve.png")
