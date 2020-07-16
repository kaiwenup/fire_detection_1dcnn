#参考博客
#https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf
#https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0
#Version 1.0
#Compatibility layer between Python 2 and Python 3
from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


# %%
#规范化数据
def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

#显示 confusion matrix
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

#打印传入的原始数据  显示的行数由preview_rows确定
def show_basic_dataframe_info(dataframe,
                              preview_rows=20):

    """
    This function shows basic information for the given dataframe
    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview
    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())

#读取传入的txt文件，做相关处理，并添加相应的表头
def read_data(file_path):

    """
    This function reads the accelerometer data from a file
    Args:
        file_path: URL pointing to the CSV file
    Returns:
        A pandas dataframe
    """

    column_names = ['user-id',
                    'activity',
                    'timestamp',  #时间戳
                   # 'co',
                   # 'smog',
                   # 't',
                    'co-fli',
                    'smog-fli',
                    't-fli'
                    ]
    df = pd.read_csv(file_path,
                     header=None, #指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
                     names=column_names)
                    #name 添加表头
    # Last column has a ";" character which must be removed ...
    df['t-fli'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    #替换操作，将‘z-axis’中后的;去掉。
    # inplace=True：改变原数据而不是改变副本的数据  regex=True：正则替换
    # ... and then this column must be transformed to float explicitly
    df['t-fli'] = df['t-fli'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    #default 0指行,1为列;
    # {‘any’, ‘all’}, default ‘any’指带缺失值的所有行;'all’指清除全是缺失值的行

    return df

#转化为浮点数
def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan


# Not used right now
def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

#图表显示
def plot_axis(ax, x, y, title):

    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(True)     #设置x轴坐标轴不可见
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])   #限制显示的范围
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

#图表显示
def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,    #将一个figure分成3个子图，分别显示
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['co-fli'], 'CO-Filtering')   #自定义函数
    plot_axis(ax1, data['timestamp'], data['smog-fli'], 'Smog-Filtering')
    plot_axis(ax2, data['timestamp'], data['t-fli'], 'Temperature-Filtering')
    plt.subplots_adjust(hspace=0.2)  
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    #subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    #其中left、bottom、right、top围成的区域就是子图的区域。wspace、hspace分别表示子图之间左右、上下的间距。

#接收read_data()处理好的txt文件里面的数据，转化为（reshape）cnn能够识别的数据帧
def create_segments_and_labels(df, time_steps, step, label_name):

    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels
    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['co-fli'].values[i: i + time_steps]
        ys = df['smog-fli'].values[i: i + time_steps]
        zs = df['t-fli'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        #寻找出现最多的标签
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

#整个数据集中，所有的数据标签
LABELS = ["fire",
          "nofire",
          ]


# The number of steps within one time segment
#sliding window 的长度
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
#sliding window 每次移动的步长
STEP_DISTANCE = 40

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('fire_data/fire_data_raw.txt')


# Describe the data
#图表显示

show_basic_dataframe_info(df, 20)

df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')

plt.show()

df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()

for activity in np.unique(df["activity"]):   #对六个行为进行遍历，并显示出部分特征数据
    #subset = df[df["activity"] == activity][:180]  #原数据是20hz采样频率，所以显示180个数据也就是显示了9s的数据（1/20 × 180 = 9）
    subset = df[df["activity"] == activity][:550]
    plot_activity(activity, subset) #自己定义的画图函数

# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
#标准化标签，将标签值统一转换成range(标签值个数-1)范围内
# Add a new column to the existing DataFrame with the encoded values
#将数据中的字符标签转化为数字标签
df[LABEL] = le.fit_transform(df["activity"].values.ravel())

# %%



print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
#区分训练集和测试集
#标号为1的为训练集，标号为2的为测试集
df_test = df[df['user-id'] > 1]
df_train = df[df['user-id'] <= 1]


# Normalize features for training data set
#**************
#**************
#**************
df_train['co-fli'] = feature_normalize(df['co-fli'])  #自定义函数  数据规范化
df_train['smog-fli'] = feature_normalize(df['smog-fli'])
df_train['t-fli'] = feature_normalize(df['t-fli'])
# Round in order to comply to NSNumber from iOS
df_train = df_train.round({'co-fli': 3, 'smog-fli': 3, 't-fli': 3})  #保留3位小数点

# Reshape the training data into segments
# so that they can be processed by the network
# x_train的数据为80×3的二维矩阵，y_train的数据为x_train数据对应的标签
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

# %%

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (20869, 40, 3)
print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y dataq
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
#num_time_periods:TIME_PERIODS x_train的行数  num_sensors:传感器个数 x_train列数
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
#打印数据中所有的标签
print(list(le.classes_))

# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
# keras不支持多维矩阵的输入，所以要将80×3的二维矩阵转化为长度为240的一维矩阵
input_shape = (num_time_periods*num_sensors)  #80×3
x_train = x_train.reshape(x_train.shape[0], input_shape)


print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)


# Convert type for Keras otherwise Keras cannot process the data
# 转化为keras能够识别的float32类型
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding：待定概念
# One-hot encoding of y_train labels (only execute once!只需执行一次)
# 待定，可能是x_train转换完成之后，y_train也要完成相应的转换
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%



print("\n--- Create neural network model ---\n")

# 1D CNN neural network
#运用Keras一维卷积实现
model_m = Sequential()
#输入数据（待定）
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
#第一次卷积层 输入矩阵大小：80×3  filter size 10  输出矩阵大小：71×100
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
#第二次卷积层  输入矩阵大小：71×100 filter size 10  输出矩阵大小：62×100
model_m.add(Conv1D(100, 10, activation='relu'))
#最大值池化层 输入矩阵大小：62×100 stride size（步长）：3
#输出矩阵：20×100
model_m.add(MaxPooling1D(3))
#第三次卷积&第四次卷积  输出2×160
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
#平均值池化层  输出1×160
model_m.add(GlobalAveragePooling1D())
#Dropout
#为减少过度拟合，部分数据被随机置0,在这里设置的为0.5（50%的数据被随机置0）
model_m.add(Dropout(0.5))
#fully connected layer
#使用softmax的激励函数
#输入1×160 输出1×6
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())
# Accuracy on training data: 99%
# Accuracy on test data: 91%



# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='h5/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
#待定  400 records  400个二维80×3二维矩阵的数据
BATCH_SIZE = 400
#训练次数
EPOCHS = 10

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,  
                      # 在训练集的数据中将数据以8：2的比例分开（2/10=0.2），
                      # 8的进行训练（train），2的进行核对（validation ）
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
#训练集数据相关参数显示
plt.figure(figsize=(6, 4))
#plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
#plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%%

print("\n--- Check against test data ---\n")

# Normalize features for training data set
#神经网络已经训练好，接下来是拿测试集进行测试。
df_test['co-fli'] = feature_normalize(df_test['co-fli'])
df_test['smog-fli'] = feature_normalize(df_test['smog-fli'])
df_test['t-fli'] = feature_normalize(df_test['t-fli'])

df_test = df_test.round({'co-fli': 3, 'smog-fli': 3, 't-fli': 3})

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

#测试集的参数（accuracy，loss）的显示
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])


# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)
#训练集的confusion matrix的显示
show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))