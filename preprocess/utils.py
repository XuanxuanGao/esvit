def select_1000():
    ##深度学习过程中，需要制作训练集和验证集、测试集。

    import os, random, shutil

    fileDir = './image/'
    tarDir = './test_image/'

    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    targetnumber = 1000
    rate = targetnumber/filenumber  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(len(sample))
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)


def check_1000():
    import os, random, shutil
    tarDir = './test_image/'
    pathDir = os.listdir(tarDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    print(filenumber)

def generate_input_names(dir_name):
    import os
    image_names = os.listdir(dir_name)
    # print(dir_name)
    # print(names)
    image_names = [os.path.join(dir_name, x)+'\n' for x in image_names]
    with open(dir_name+'.txt', mode='w') as f:
        f.writelines(image_names)


if __name__ == "__main__":
    # select_1000()
    # check_1000()
    generate_input_names(dir_name = 'image_Resize_100')