# encoding:utf-8
#神经网络实现数字识别
import struct
import sys
from datetime import datetime

from fcn import Network


def transpose(param):
    pass


class Loader(object):
    #数据基础加载器
    def __init__(self,path,number):
        '''

        :param path: 数据文件路经
        :param number: 文件中样本个数
        '''
        self.path = path;
        self.number = number;

    def get_file_content(self):
        file = open(self.path,'rb')
        content = file.read()
        file.close();
        return content;

    def to_int(self,byte):
        #将unsigned char转成int
        return struct.unpack('B',byte)[0]



class ImgaeLoader(Loader):
#这里的self即指向ImageLoader基类自身也指向Loader父类
    def load(self):
        #首先得到文件内容
        content = self.get_file_content()
        data_set = []
        for i in range(self.number):
            data_set.append(self.get_one_sample(self.get_picture(content,i)))
        return data_set

    def get_picture(self, content, index):
        #获得图片二维矩阵
        #t=图片样本前16位不是标签所以要从第17位开始（content[16])
        start = index * 28 * 28 + 16;
        picture = [];
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start + i * 28 + j]))
            #     if self.to_int(content[start + i * 28 + j]) != 0:
            #         sys.stdout.write('*')
            #     else:
            #         sys.stdout.write(' ')
            # print ' '
        return picture;

    def get_one_sample(self, picture):
        #将图片二维矩阵转为一维输入向量
        sample = [];
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample;


class LabelLoder(Loader):
    def load(self):
        #标签前8位不是标签所以要从第9位开始（content[8])
        content = self.get_file_content();
        label_set = [];
        for i in range(self.number):
            label_set.append(self.norm(content[i + 8]))
        return label_set;

    def norm(self,label):
        #将label值一个0到9的数字转为一个10维的输出向量
        label_vec = [];
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_train_data_set():
    image_loader = ImgaeLoader('train-images.idx3-ubyte',60)
    label_loader = LabelLoder('train-labels.idx1-ubyte',60)
    return image_loader.load(),label_loader.load();

def get_test_data_set():
    image_loader = ImgaeLoader('t10k-images.idx3-ubyte',10)
    label_loader = LabelLoder('t10k-labels.idx1-ubyte',10)
    return image_loader.load(),label_loader.load()


def now():
    return datetime.now().strftime("%c")


def get_result(input):
    max_value = - 1;
    max_index = -1;
    for i in range(len(input)):
        if input[i] > max_value:
            max_index = i;
            max_value = input[i]
    return max_index;

def evaluate(net, test_data_set, test_labels):
    error = 0;

    for i in range(len(test_labels)):
        #样本和预测输出的虽然都是10维的一个向量，但是不能直接比较相等来代表预测是否正确
        #因为预测值的10维向量不是0.1,0.9组成的，只能通过最大值的位置作为结果来比较
        label = get_result(test_labels[i])
        predict = get_result(net.predict(test_data_set[i]))
        if label != predict:
            error += 1;
    return float(error) / float(len(test_labels))

def train_and_evaluate():
    epoch = 1;
    last_error_ratio = 1.0;
    #获取数据集并处理
    # train_data_set,tain_labels = transpose(get_train_data_set())
    # test_data_set,test_labels = transpose(get_test_data_set())
    train_data_set, tain_labels = get_train_data_set()
    test_data_set, test_labels = get_test_data_set()

    #搭建神经网络
    net = Network([784,300,10]);

    #训练
    while True:
        net.train(tain_labels,train_data_set,1,0.01)
        epoch += 1;
        print '%s epoch: %d   \n' % (now(),epoch)

        if epoch % 10 == 0:
            #每训练10次就通过测试样本计算错误率
            error_ratio = evaluate(net,test_data_set,test_labels)
            print '%s epoch: %d  error ratio : %f\n' % (now(),epoch,error_ratio)
            #如果错误率升高就停止训练防止过拟合，否则更新last_error_ratio为最新的错误率
            if error_ratio > last_error_ratio:
                break;
            else:
                last_error_ratio = error_ratio;

if __name__ == '__main__':
    train_and_evaluate();
    # train_data_set, tain_labels = get_train_data_set()
    # print train_data_set,'   ',tain_labels
    # a = [[1,2],[4,5]]
    # print a[-1]