# encoding:utf-8
from perception import Perception
import sys

reload(sys)

sys.setdefaultencoding('utf8')
#定义激活函数
f = lambda x: x;


def getDataset():
    #输入特征为一个即工作年限
    inputs = [[5], [3], [8], [1.4], [10.1]];
    labels = [5500, 2300, 7600, 1800, 11400];
    return inputs,labels;


class LinearUnit(Perception):
    #继承感知器实现线性单元
    def __init__(self,input_num):
        Perception.__init__(self,input_num,f);


def train_Linear_Unit():
    #设置输入参数的个数为1 即input_num=1;激活函数为f
    L = LinearUnit(1);
    inputs,labels = getDataset();
    #迭代10次
    L.train(inputs,labels,10,0.01);
    return L;


if __name__ == '__main__':
    linearUnit = train_Linear_Unit();
    print linearUnit;
    print "Worked for 3 years, monthly salary:",linearUnit.predict([3]);
    #和标签值8年7600 还是有些差距的
    print "Worked for 8 years, monthly salary:", linearUnit.predict([8]);
    print "Worked for 11 years, monthly salary:", linearUnit.predict([15]);