---
title: 基于Keras的一维卷积神经网络
date: 2020-08-06
categories: 
- 计算机
- DeepLearning
tags:
- TensorFlow
---

一维卷积神经网络：1D Convolutional Neural Networks（1D CNN）

[本项目源代码](https://github.com/kaiwenup/fire_detection_1dcnn)

本程序基于GitHub开源代码修改（[代码链接](https://github.com/ni79ls/har-keras-cnn)，[训练数据下载链接](http://www.cis.fordham.edu/wisdm/dataset.php)）

## 文件目录结构

```markdown
fire_1d_cnn
 -fire_1d_cnn.py
 -.gitignore
 -fire_data
 -h5
```

`fire_1d_cnn.py`：Python文件

`.gitignore`：git中设置无需纳入git管理的文件

`fire_data`：存放运行神经网络所需要的数据的文件夹

`h5`：存放程序运行后生成的模型的文件夹

<!--more-->

## 环境搭建

系统：Ubuntu16.04

- Python3.X
- Tensorflow2.2.0
- seaborn
- numpy
- scipy
- sklearn
- pandas
- matplotlib

### 安装步骤

#### 安装Python3

```markdown
sudo apt install python3
```

同理：pip也要对应安装的python版本，三代对应：python3-pip

#### 安装pip

```markdown
sudo apt-get install python3-pip
```

然后把pip更新到最新版本

```markdown
sudo pip3 install --upgrade pip
```

#### 验证

最后查看版本来检查是否安装好相应的版本

```markdown
python3 -V
pip3 -V
```

#### 安装TensorFlow

注意：一定要版本对应,因为我用的是pip3，所以通过`pip3 install`来下载。

可以通过官网来安装，但由于没有翻墙，所以速度奇慢。

可以使用国内源进行下载

```markdown
pip3 install tensorflow-cpu==2.2.0 -i https://pypi.douban.com/simple/
```

#### TensorFlow验证

打开Python解释器，在命令行输入

```markdown
python3
```

注意，输入`python`默认打开python2.X。

然后导入TensorFlow库：

```
import tensorflow as tf
```

请注意，如果没有正确安装TensorFlow，或者将带GPU的版本安装在不受支持的系统上，就会在此处出现错误。CUDA错误在这一步上非常普遍。如果有效，就可以试试打印出TensorFlow的版本：

```
print(tf.__version__)
```

（注意version前后有2个下划线）

完成后，应该看到打印出的TensorFlow版本 ，由于本次安装的是`2.2.0`，所以最后输出`2.2.0`。

#### 第三方库扩展库下载

```
pip3 install matplotlib
pip3 install numpy
pip3 install scipy
pip3 install pandas
pip3 install seaborn
pip3 install sklearn
pip3 install --upgrade keras 
```

如果运行程序的时候还是报错缺少第三方库，则可以按照提示用`pip3 install 第三方库名字`命令进行下载。

如果下载过慢或者下载报错,可以尝试国内的源

```markdown
#安装matplotlib 
python3 -m pip3 install matplotlib -i http://pypi.douban.com/simple --trusted-host pypi.douban.com 
#安装pandas 
python3 -m pip3 install pandas -i http://pypi.douban.com/simple --trusted-host pypi.douban.com 
#安装seaborn
python3 -m pip3 install seaborn scipy -i http://pypi.douban.com/simple --trusted-host pypi.douban.com 
```

其他第三方库也可以自行搜索国内源，下载速度会快一点。

#### 运行测试

在`fire_1d_cnn`文件夹下运行`python3 fire_1d_cnn.py `，若程序运行无报错，则环境搭建完成。

### 参考博客

- [ubuntu中安装python3和pip](https://www.cnblogs.com/litifeng/p/11107311.html)
- [ubuntu 16.4 , python3.5安装TensorFlow以及环境配置](https://blog.csdn.net/perfect1t/article/details/81017970?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-3)
- [在Ubuntu搭建TensorFlow环境](https://blog.csdn.net/chszs/article/details/78987532)
- [ubuntu16.04上tensorflow安装遇到的问题](https://blog.csdn.net/yang1994/article/details/104380415)

## CNN构建

![cnn框架](https://i.loli.net/2020/08/06/YUERexG7NDtjs8h.jpg)

### 导入第三方库

程序中一共导入了numpy、pandas、seaborn、scipy、sklearn等第三方库，如果运行时提示未下载对应库，安装提示安装下载即可。

### 导入原始数据

#### 原始数据构成

所有数据都存放在txt文档中（也可以存在csv文件中），存放路径为`fire_data/fire_data_raw.txt`，每一行都以`;`结尾

第一列数据为数据所在的组别，在本数据集中分为两个组别，分别是1和2。其中1表示训练集的数据，2表示为测试集的数据。第二列为所在列的数据标签，标识着这一列数据对应的状态，本数据集中一共有两个标签`fire`和`nofire`。第三列表示时间戳。第四、五、六列分别表示采集到的一氧化碳、烟雾以及温度数据（所有数据都做了放大处理，放大倍数为100）。

#### 导入数据

相关函数：

```python
df = read_data('fire_data/fire_data_raw.txt')

#read_data函数的定义
def read_data(file_path):
    
    column_names = ['user-id',    # 组别
                    'activity',   # 标签
                    'timestamp',  # 时间戳
                    'co-fli',     # 以下三个表头为传感器采集到的数据
                    'smog-fli',
                    't-fli'
                    ]
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    df['t-fli'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')
    df['t-fli'] = df['t-fli'].apply(convert_to_float)
    df.dropna(axis=0, how='any', inplace=True)

    return df
```

该函数的作用是为数据添加表头，然后将每一行末尾的`;`用正则表达式替换成` `（空格）。最后去掉带缺失值的所有行。

具体是哪一行实现的见代码注释。

### 原始数据可视化

数据可视化主要是将海量数据更加直观的展现出来，程序中一共使用了柱状图和折线图两种形式。`dis_switch`表示图表显示的开关。

```Python
show_basic_dataframe_info(df, 20)
```

显示原始数据的前20行数据。↑

```python
    df['activity'].value_counts().plot(kind='bar',
                                    title='Training Examples by Activity Type')
    plt.show()

    df['user-id'].value_counts().plot(kind='bar',
                                    title='Training Examples by User')
    plt.show()
```

柱状图显示↑

```python
for activity in np.unique(df["activity"]):   
    subset = df[df["activity"] == activity][:550]
    if dis_switch:
        plot_activity(activity, subset) 
```

折线图显示↑

## 数据处理

### 标签处理

由于现在的标签是以字符串的形式出现的，需要转化为数字形式，实现代码：

```python
LABEL = "ActivityEncoded"
le = preprocessing.LabelEncoder()
df[LABEL] = le.fit_transform(df["activity"].values.ravel())
```

### 数据归一化

由于采集到的三种传感器数据，量纲不同，所以要对数据进行归一化处理。在数据挖掘数据处理过程中，不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。

![image-20200807193440321](https://i.loli.net/2020/08/07/VmGpYEqoS4ruw15.png)

其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

代码实现如下：


```python
def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)   # axis = 0:压缩行，对各列求均值，返回 1* n 矩阵
    sigma = np.std(dataset, axis=0) # axis = 0:计算每一列的标准差  
    return (dataset - mu)/sigma
```


参考博客：

- [中心化（又叫零均值化）和标准化（又叫归一化）](https://blog.csdn.net/GoodShot/article/details/80373372)
- [数据预处理之中心化（零均值化）与标准化（归一化）](https://www.cnblogs.com/wangqiang9/p/9285594.html)

### 数据类型转换

keras只识别`float32`类型的数据，所以要进行类型转换。

代码实现如下：

```python
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")
```

### 数据切块重组^*^

从txt导入的数据并不能直接导入1DCNN训练模型，需要将数据进行重组`reshape`

导入1DCNN的数据只需要标签以及三个传感器的数据，也就是原始数据的其中四列数据，其他数据都可以暂时弃掉。

留下来的传感器数据以每80行作为一个片段（`segment`），同时将这80行中出现最多的标签作为该片段的标签。最后每隔40行（步长）截取一个片段。

片段的长度以及步长都可以进行设置，参数设置代码如下：

```python
TIME_PERIODS = 80
STEP_DISTANCE = 40
```

![数据重组](https://i.loli.net/2020/08/07/bSaE98dTevuPJpG.jpg)

这样原始数据就可以切成很多片段，每个片段都是一个80×3的矩阵以及相应的标签组成。

代码实现如下：

```python
def create_segments_and_labels(df, time_steps, step, label_name):
    N_FEATURES = 3
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['co-fli'].values[i: i + time_steps]
        ys = df['smog-fli'].values[i: i + time_steps]
        zs = df['t-fli'].values[i: i + time_steps]
        #寻找出现最多的标签
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels
```

```python
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)
```

数据切块之后，将传感器数据存到`x_train`中，将每一个片段的标签存到`y_train`中

切块后的数据还是不能直接导入keras进行训练，因为keras不支持多维矩阵的输入（不确定，后续可以测试一下是否支持多维矩阵输入），所以还需要将多维矩阵转化为1\*240的一维矩阵，也就是长度为240的数组。

代码实现如下：

```python
input_shape = (num_time_periods*num_sensors)  # 80×3
x_train = x_train.reshape(x_train.shape[0], input_shape)
```

### 划分训练集和测试集

训练集数据只使用在搭建神经网络的过程中，而测试集数据只在神经网络搭建后用于评估神经网络模型的正确度时使用。若将训练集数据用于评估会对最后的测试结果造成影响。

本项目中使用数据集中的组别也就是`user-id`来划分训练集和测试集

```Python
df_test = df[df['user-id'] > 1]
df_train = df[df['user-id'] <= 1]
```

其中组别为小于等于1的为训练集，大于1的为测试集

## 搭建1DCNN模型^*^

![1_Y117iNR_CnBtBh8MWVtUDg](https://i.loli.net/2020/08/07/lwHn4cFbRq1fdjk.png)

[图片来源](https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf)

**输入数据：** 数据经过预处理后，每条数据记录中包含有 80 个时间片（这样就得到了一个 80 x 3 的矩阵。然后所有数据平展成长度为 240 的向量后传入神经网络中）。网络的第一层必须再将其变形为原始的 80 x 3 的形状。

**第一个 1D CNN 层：** 第一层定义了高度为 10（也称为卷积核大小）的滤波器`filter`（也称为特征检测器`feature detector`）。只有定义了一个滤波器，神经网络才能够在第一层中学习到一个单一的特征。这可能还不够，因此我们会定义 100 个滤波器。这样我们就在网络的第一层中训练得到 100 个不同的特性。第一个神经网络层的输出是一个 71 x 100 的矩阵。输出矩阵的每一列都包含一个滤波器的权值。在定义内核大小并考虑输入矩阵长度的情况下，每个过滤器将包含 71 个权重值。

**第二个 1D CNN 层：** 第一个 CNN 的输出结果将被输入到第二个 CNN 层中。我们将在这个网络层上再次定义 100 个不同的滤波器进行训练。按照与第一层相同的逻辑，输出矩阵的大小为 62 x 100。

**最大值池化层：** 为了减少输出的复杂度和防止数据的过拟合，在 CNN 层之后经常会使用池化层。在我们的示例中，我们选择了大小为 3 的池化层。这意味着这个层的输出矩阵的大小只有输入矩阵的三分之一。

**第三和第四个 1D CNN 层：** 为了学习更高层次的特征，这里又使用了另外两个 1D CNN 层。这两层之后的输出矩阵是一个 2 x 160 的矩阵。

**平均值池化层：** 多添加一个池化层，以进一步避免过拟合的发生。这次的池化不是取最大值，而是取神经网络中两个权重的平均值。输出矩阵的大小为 1 x 160 。每个特征检测器在神经网络的这一层中只剩下一个权重。

**Dropout 层：** Dropout 层会随机地为网络中的神经元赋值零权重。由于我们选择了 0.5 的比率，则 50% 的神经元将会是零权重的。通过这种操作，网络对数据的微小变化的响应就不那么敏感了。因此，它能够进一步提高对不可见数据处理的准确性。这个层的输出仍然是一个 1 x 160 的矩阵。

**使用 Softmax 激活的全连接层：** 最后一层将会把长度为 160 的向量降为长度为2 的向量，因为我们有2个类别要进行预测（即 "fire"、"nofire"）。这里的维度下降是通过另一个矩阵乘法来完成的。Softmax 被用作激活函数。它强制神经网络的所有六个输出值的加和为一。因此，输出值将表示这六个类别中的每个类别出现的概率。
[参考博客](https://juejin.im/post/6844903713224523789)

代码如下：

```python
# 1D CNN neural network
# 运用Keras一维卷积实现
model_m = Sequential()
# 输入数据（待定）
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# 第一次卷积层 输入矩阵大小：80×3 输出矩阵大小：71×100
# kernel/patch size:10 filter size:100
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# 第二次卷积层  输入矩阵大小：71×100 输出矩阵大小：62×100
# kernel/patch size:10 filter size:100
model_m.add(Conv1D(100, 10, activation='relu'))
# 最大值池化层 输入矩阵大小：62×100 stride size（步长）：3
# 输出矩阵：20×100
model_m.add(MaxPooling1D(3))
# 第三次卷积 输入矩阵大小：20×160 输出11×160
# kernel/patch size:10 filter size:160 
model_m.add(Conv1D(160, 10, activation='relu'))
# 第四次卷积 输入矩阵大小：11×160 输出2×160
model_m.add(Conv1D(160, 10, activation='relu'))
# 平均值池化层  输出1×160
model_m.add(GlobalAveragePooling1D())
# Dropout层
# 为减少过度拟合，部分数据被随机置0,在这里设置的为0.5（50%的数据被随机置0）
model_m.add(Dropout(0.5))
# fully connected layer
# 使用softmax的激励函数
# 输入1×160 输出1×6
model_m.add(Dense(num_classes, activation='softmax'))
```

### 参考博客：

- [[译] 在 Keras 中使用一维卷积神经网络处理时间序列数据](https://juejin.im/post/6844903713224523789)
- [Introduction to 1D Convolutional Neural Networks in Keras for Time Sequences](https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf)
- [Human Activity Recognition (HAR) Tutorial with Keras and Core ML](https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0)
- [Understanding Convolutional Neural Networks for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

## 模型拟合(Fit the model)

该过程就是开始训练神经网络。

代码如下：

```python
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='h5/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 400
EPOCHS = 50

history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
```

Epoch：所有的数据集都全部通过神经网络模型时算一次Epoch

Batch：批，神经网络中都是分批次处理数据的。

Batch_size：每批处理的样本的个数。

Number of batchs：批的数量

举个栗子：

假设数据集一共有2000个样本，每一批有400个样本则：

![概念图示](https://i.loli.net/2020/08/07/uht6DS2FC5EQvxr.jpg)

validation_split：在训练集的数据中将数据以8：2的比例分开（2/10=0.2）（8的进行训练（train），2的进行验证（validation ））

也就是说上文提到的训练集数据在这里又被拆解成训练集和验证集。

训练集，验证集以及测试集的区别见参考博客。

每次训练完成，都生成了后缀名为`.h5`的文件。该文件为神经网络模型，可以在其他文件中调用该模型，无需每次生成。

### 参考博客

- [机器学习中常见的概念Epoch、Batch Size、Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
- [训练集、验证集、测试集的区别与应用](https://blog.csdn.net/qq_35082030/article/details/83544593)

## 训练结果可视化

```python
plt.figure(figsize=(6, 4))
#plt.plot(history.history['acc'], "g--", label="Accuracy of training data")         # 版本不同传入不同的参数
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
```

可能由于版本的不同`history.history`传入的参数不一样，如果报错，试试注释掉的代码。

训练结果如图示：

![Figure_5](https://i.loli.net/2020/08/07/gMBP9rLkY2pCbdI.png)

横坐标训练次数，纵坐标准确率和错误率

Accuracy of train data:  训练集数据准确率

Accuracy of validation data: 评估集数据准确率

Loss of train data: 训练集数据错误率

Lossof validation data: 评估集数据错误率

## 测试集测试

模型训练完成之后，需要用测试集数据测试生成神经网络模型的准确性。

```python
y_pred_test = model_m.predict(x_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)
show_confusion_matrix(max_y_test, max_y_pred_test)
print(classification_report(max_y_test, max_y_pred_test))
```

最后用测试集得到的测试结果如下：

![Figure_4](https://i.loli.net/2020/08/07/6NeUjktQLao3JSd.png)

其中横坐标为神经网络模型预测的每个输入数据块对应的标签，纵坐标为实际的数据块对应的标签。

举个栗子：

矩阵第一行的所有数字之和为所有带有`fire`标签的数据块数量，也就是一共425个数据块。现在只看第一行，神经网络模型预测的带`fire`标签的数据块为425个，标签为`nofire`的数据块格式为0。也就是说准确率100%。