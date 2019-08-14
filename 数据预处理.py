#--coding--:utf-8
import tensorflow as tf
import matplotlib.pyplot as plt

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile('images/plate1.jpg', 'rb').read()

# 使用pyplot显示图像
def show(img_data):
    plt.imshow(img_data.eval())
    plt.show()


#####################################################
# # 1、对 JPEG 格式图像的解码处理
# with tf.Session() as sess:
#     # 将原始数据解码成多维矩阵
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     print(img_data.eval())
#     show(img_data)
#
#     # 将图像的矩阵编码成图像并存入文件
#     encoded_image = tf.image.encode_jpeg(img_data)
#     with tf.gfile.GFile('output.jpg', 'wb') as f:
#         f.write(encoded_image.eval())
#
#     # 将图像数据的类型转为实数类型，便于对图像进行处理
#     img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
#####################################################


#####################################################
# 2、大小调整
# 用resize_images调整图像大小
# 第一个参数为原始图像
# 第二个参数为调整后的图像大小[new_height,new_width]，跟旧版本分为两个参数不一样
# method参数给出了调整图像大小的算法
with tf.Session() as sess:
    # 将原始数据解码成多维矩阵
    img_data = tf.image.decode_jpeg(image_raw_data)
    # print(img_data.eval())
    print("原始图像:")
    show(img_data)

    # 将图像的矩阵编码成图像并存入文件
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('output.jpg', 'wb') as f: # 获取文本操作句柄，类似于python提供的文本操作open()函数，
                                                  # filename是要打开的文件名，mode是以何种方式去读写，将会返回一个文本操作句柄。
        f.write(encoded_image.eval())

    # 将图像数据的类型转为实数类型，便于对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    resized = tf.image.resize_images(img_data, [300, 300], method=0)  # method=0代表双线性插值
    # print(resized.get_shape())  # 图像深度没有显式指定则为问号
    print("裁剪后的图像:")
    show(resized)

    # 用resize_image_with_crop_or_pad调整图像大小
    # 第一个参数为原始图像
    # 第二个和第三个参数是调整后的图像大小，大于原图则填充，小于则裁剪居中部分
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 200, 200)
    print("通过resize_image_with_crop_or_pad处理后的图像，此图为裁剪后:")
    show(croped)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 500, 700)
    print("通过resize_image_with_crop_or_pad处理后的图像，此图为填充后:")
    show(padded)

    # 用central_crop调整图像大小
    # 第一个参数是原始图像
    # 第二个参数为调整比例，是(0,1]的实数
    central_cropped = tf.image.central_crop(img_data, 0.5)
    print("通过比例裁剪后的图像:")
    show(central_cropped)
#####################################################


#####################################################
# 3、图像翻转
# with tf.Session() as sess:
#     # 将原始数据解码成多维矩阵
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # print(img_data.eval())
#     print("原始图像:")
#     show(img_data)
#
#     # 图像翻转
#     flipped = tf.image.flip_up_down(img_data)  # 上下
#     print("上下翻转后:")
#     show(flipped)
#     flipped = tf.image.flip_left_right(img_data)  # 左右
#     print("左右翻转后:")
#     show(flipped)
#     transposed = tf.image.transpose_image(img_data)  # 对角线
#     print("沿对角线反转后: ")
#     show(transposed)
#
#     # 随机翻转图像
#     flipped = tf.image.random_flip_up_down(img_data)  # 随机上下
#     print("随机上下翻转后:")
#     show(flipped)
#     flipped = tf.image.random_flip_left_right(img_data)  # 随机左右
#     print("随机左右翻转后:")
#     show(flipped)
#
#
# ### 色彩调整
# with tf.Session() as sess:
#     # 将原始数据解码成多维矩阵
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     # print(img_data.eval())
#     print("原始图像:")
#     show(img_data)
#
#     # 调整图像的亮度
#     adjusted = tf.image.adjust_brightness(img_data, 0.5)  # 将图像的亮度+0.5
#     print("将图像的亮度+0.5:")
#     show(adjusted)
#     adjusted = tf.image.random_brightness(
#         img_data, max_delta=0.5)  # 在[-0.5,0.5]范围内随机调整图像亮度
#     print("在[-0.5,0.5]范围内随机调整图像亮度:")
#     show(adjusted)
#
#     # 调整图像的对比度
#     adjusted = tf.image.adjust_contrast(img_data, -5)  # 将图像的对比度-5
#     print("将图像的对比度-5:")
#     show(adjusted)
#     adjusted = tf.image.adjust_contrast(img_data, 5)  # 将图像的对比度+5
#     print("将图像的对比度+5:")
#     show(adjusted)
#     adjusted = tf.image.random_contrast(img_data, lower=1, upper=5)  # 随机调整对比度
#
#     # 调整图像的饱和度
#     adjusted = tf.image.adjust_saturation(img_data, -5)  # 将饱和度-5
#     print("将饱和度-5:")
#     show(adjusted)
#     adjusted = tf.image.adjust_saturation(img_data, 5)  # 将饱和度+5
#     print("将饱和度+5:")
#     show(adjusted)
#     adjusted = tf.image.random_saturation(img_data, lower=1, upper=5)  # 随机调整饱和度
#
#     # 调整图像的色相
#     adjusted = tf.image.adjust_hue(img_data, 0.5)  # 将色相+0.5
#     print("将色相+0.5:")
#     show(adjusted)
#     adjusted = tf.image.random_hue(img_data, max_delta=0.5)  # 随机调整色相
#     print("随机调整色相:")
#     show(adjusted)
#####################################################