

### intelSceneClassification_FlyAI

### [![GPL LICENSE](https://badgen.net/badge/License/GPL/green)](https://www.gnu.org/licenses/gpl-3.0.zh-cn.html) [![GPL LICENSE](https://badgen.net/badge/Supported/TensorFlow,Keras,PyTorch/green?list=1)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Python/3.+/green)](https://flyai.com) [![GPL LICENSE](https://badgen.net/badge/Platform/Windows,macOS,Linux/green?list=1)](https://flyai.com)

### [项目官方网址](https://www.flyai.com/d/intelSceneClassification)

> 通过迭代自己的算法模型并提交训练，获取奖金池高额奖金。

***

#### 竞赛说明

- **参加项目竞赛必须实现 `model.py` 中的`predict_all`方法，系统才能给出最终分数。**

#### 样例代码说明

##### `app.yaml`

> 是项目的配置文件，项目目录下**必须**存在这个文件，是项目运行的依赖。

##### `processor.py`

> **样例代码中已做简单实现，可供查考。**
>
> 处理数据的输入输出文件，把通过csv文件返回的数据，处理成能让程序识别、训练的矩阵。
>
> 可以自己定义输入输出的方法名，在`app.yaml`中声明即可。
>
> ```python
>     def input_x(self, image_path):
>         '''
>         参数为csv中作为输入x的一条数据，会被Dataset类中的next_batch()方法多次调用。
>         :params: 输入的数据列表
>         :return: 返回矩阵
>         '''
>         pass
> 
>     def input_y(self, label):
>         '''
>         参数为csv中作为输入y的一条数据，会被Dataset类中的next_batch()方法多次调用。
>         :params: 数据标签列表
>         :return: 返回矩阵
>         '''
>         pass
>     
>     def output_y(self, data):
>         '''
>         验证时使用，把模型输出的y转为对应的结果
>         :param data: 预测返回的数据
>         :return: 返回预测的标签
>         '''
>         pass
> 
> ```

##### `main.py`

> **样例代码中已做简单实现，可供查考。**
>
> 程序入口，编写算法，训练模型的文件。在该文件中实现自己的算法。
>
> 通过`dataset.py`中的`next_batch`方法获取训练和测试数据。
>
> ```python
> # 数据获取辅助类
> dataset = Dataset()
> x_train, y_train, x_test, y_test = dataset.next_batch(BATCH)
> ```
>
> 通过`model.py`中的`save_model`方法保存模型
>
> ```python
> # 模型操作辅助类
> model = Model(dataset)
> model.save_model(YOU_NET)
> ```
>
> **如果使用`PyTorch`框架，需要在`net.py`文件中实现网络。其它用法同上。**

##### `model.py`

> **样例代码中已做简单实现，可供查考。**
>
> 训练好模型之后可以继承`flyai.model.base`包中的`base`重写下面三个方法实现模型的保存、验证和使用。
>
> ```python
>    def predict(self, **data):
>         '''
>         使用模型
>       	:param data: 模型的输入的一个或多个参数
>         :return:
>         '''
>         pass
> 
>     def predict_all(self, datas):
>         '''
>         （必须实现的方法）评估模型，对训练的好的模型进行打分
>       	:param datas: 验证集上的随机数据，类型为list
>         :return outputs: 返回调用模型评估之后的list数据
>         '''
>         pass
>    
>     def save_model(self, network, path=MODEL_PATH, name=MODEL_NAME, overwrite=False):
>         '''
>         保存模型
>         :param network: 训练模型的网络
>         :param path: 要保存模型的路径
>         :param name: 要保存模型的名字
>         :param overwrite: 是否覆盖当前模型
>         :return:
>         '''
>         self.check(path, overwrite)
> 
> ```

##### `predict.py`

>**样例代码中已做简单实现，可供查考。**
>
>对训练完成的模型使用和预测。

##### `path.py`

> 可以设置数据文件、模型文件的存放路径。

##### `dataset.py`

> 该文件在**FlyAI开源库**的`flyai.dataset`包中，通过`next_batch(BATCH)`方法获得`x_train` `y_train` `x_test` `y_test`数据。
>
> FlyAI开源库可以通过`pip3 install -i https://pypi.flyai.com/simple flyai` 安装。

***

#### FlyaI终端命令

##### Windows用户

##### 1. 下载项目并解压

#####  2.打开运行，输入cmd，打开终端

> Win+R 输入cmd

##### 3. 使用终端进入到项目的根目录下

首先进入到项目对应的磁盘中，然后执行

> cd path\to\project
>
> Windows用户使用 flyai.exe

##### 4. 初始化环境并登录

下载完成之后，执行下列命令并使用微信扫码登录

> flyai.exe init

登录成功之后，会自动下载运行所需环境

##### 5. 本地开发调试

执行

> flyai.exe test

安装项目所需依赖，并运行 main.py

如果使用本地IDE开发，可以自行安装 requirements.txt 中的依赖，运行 main.py 即可

##### 6.提交训练到GPU

项目中如有新的引用，需加入到 requirements.txt 文件中

在终端下执行

> flyai.exe train

返回sucess状态，代表提交离线训练成功

默认训练成功后不公开在项目排行榜中，公开项目需在提交训练时执行

> flyai.exe train -p=1

完整训练设置执行代码示例：

> flyai.exe train -p=1 -b=32 -e=100

通过执行训练命令，本次训练循环 100 次，每次训练读取的数据量为 32 ，公开提交模型

##### Mac或Linux用户

##### 1. 下载项目并解压

##### 2. 使用终端进入到项目的根目录下

> cd /path/to/project
>
> Mac和Linux用户使用 ./flyai 脚本文件

##### 3. 初始化环境并登录

授权flyai脚本

> chmod +x ./flyai

下载完成之后，执行下列命令并使用微信扫码登录

> ./flyai init

登录成功之后，会自动下载运行所需环境

##### 4. 本地开发调试

执行

> ./flyai  test

安装项目所需依赖，并运行 main.py

如果使用本地IDE开发，可以自行安装 requirements.txt 中的依赖，运行 main.py 即可

##### 5.提交训练到GPU

项目中如有新的引用，需加入到 requirements.txt 文件中

在终端下执行

> ./flyai train

返回sucess状态，代表提交离线训练成功

训练结束会以微信和邮件的形式发送结果通知

默认训练成功后不公开在项目排行榜中，公开项目需在提交训练时执行

> ./flyai train -p=1

完整训练设置执行代码示例：

> ./flyai train -p=1 -b=32 -e=100

通过执行训练命令，本次训练循环 100 次，每次训练读取的数据量为 32 ，公开提交模型

***


### [FlyAI全球人工智能专业开发平台，一站式服务平台](https://flyai.com)

**扫描下方二维码，及时获取FlyAI最新消息，抢先体验最新功能。**



[![GPL LICENSE](https://www.flyai.com/images/coding.png)](https://flyai.com)



