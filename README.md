# TensorFlow_Practice  -  基于TensorFlow的代码实现
自注：本书是以TensorFlow 1.0.0-rc0作为示例讲解

MNIST数据集下载：


Cifar 10数据集和相关py下载：
1、git clone https://github.com/tensorflow/models.git  找到models/tutorials/image/cifar10，把cifar10.py和cifar10_input.py拷到此目录下
2、pip install tensorflow-datasets
3、因为目前已经没有cifar10.maybe_download_and_extract()这个方法了，该方法已经集成在了cifar10_input.distorted_inputs(batch_size=batch_size)中；因此直接调用这个方法，会自动下载，下载地址我这里是C:\Users\Administrator\tensorflow_datasets\cifar10\1.0.2... ；另外有网友说可能因为网络问题报错误500，这时去掉DATA_URL中https的s即可，参考https://www.lizenghai.com/archives/10386.html）
如果还不行，直接在上面git clone下载下来的文件夹内搜索找到cifar10_download_and_extract.py这个文件，然后直接执行，就会下载cifar10数据集了，速度很快；然后拷贝到
C:\Users\Administrator\tensorflow_datasets\downloads\extracted下