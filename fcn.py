# coding:utf-8
# 全连接神经网络的实现
import random
from math import e

from numpy import exp


def sigmoid(output):
    return 1 / (1 + exp(-output))

#结点类 负责记录和维护结点自身信息以及这个结点的上下游连接 实现输出值和误差项的计算
class Node(object):
    #layer_index 层编号 node_index 结点编号
    def __init__(self,layer_index,node_index):
        self.layer_index = layer_index;
        self.node_index = node_index
        self.upstream = [];
        self.downstream = [];
        self.output = 0.0;
        self.delta = 0.0;
    #结点绑定连接
    def append_upstream_connection(self,conn):
        #增加一个到上游结点的连接
        self.upstream.append(conn)
    def append_downstream_connection(self,conn):
        #增加一个到下游结点连接
        self.downstream.append(conn)

    def calc_output(self):
        #根据式1计算
        # output = reduce(map(lambda up_conn:up_conn.upstream_node.output*up_conn.weight,self.upstream),0.0)
        output = reduce(lambda sum, up_conn: sum+up_conn.upstream_node.output*up_conn.weight,self.upstream,0.0)
        self.output = sigmoid(output)
    def calc_output_layer_delta(self,label):
        #根据式3计算
        self.delta = self.output * (1 - self.output)*(label - self.output)

    def calc_hidden_layer_delta(self):
        #根据式4 e4=a4*(1-a4)(w84e8 + w94e9) e为结点误差项
        temp = reduce(lambda sum,down_conn: sum + down_conn.weight * down_conn.downstream_node.delta,self.downstream,0.0)
        self.delta = self.output * (1 - self.output) * temp;

    def __str__(self):
        node_str = "%u-%u: output: %f, delta: %f" % (self.layer_index,self.node_index,self.output,self.delta)
        downstream_str = reduce(lambda sum, down_conn:sum + '\n\t' + str(down_conn),self.downstream,' ')
        upstream_str = reduce(lambda sum, up_conn: sum + '\n\t' + str(up_conn),self.upstream,' ')
        print node_str,'\n\tdownstream: ',downstream_str,'\n\tupstream: ', upstream_str;
class ConstNode(object):
    def __init__(self,layer_index,node_index):
        self.layer_index = layer_index
        self.node_index = node_index;
        self.downstream = [];
        self.output = 1;

    def append_downstream_connection(self,conn):
        #增加一个到下游结点的连接
        self.downstream.append(conn)

    def __str__(self):
        node_str = '%u-%u output: %f' % (self.layer_index,self.node_index,self.output)
        downstream_str = reduce(lambda sum, down_conn: sum + '\n\t' + str(down_conn), self.downstream, ' ')
        return node_str, '\n\tdownstream: ', downstream_str


class Layer(object):
    # layer负责初始化　作为Node的集合对象,提供对Node的集合操作
    def __init__(self,layer_index,node_number):
        '''

        :param layer_index: 神经网络层编号
        :param node_number:层结点编号
        '''
        self.layer_index = layer_index;
        self.nodes = [];
        for i in range(node_number):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,i))

    def set_output(self, sample):
        for i in range(len(sample)):
            self.nodes[i].output = sample[i];

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print node


class Conntection(object):
    def __init__(self,upstream_node,downstream_node):
        '''
        初始化连接 权重初始化为一个很小的随机数
        :param upstream_node:连接的上游结点
        :param downstream_node:连接的下游结点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1,0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        #梯度为式5 rate后面的乘积 即e*x 误差项乘输入
        self.gradient = self.downstream_node.delta * self.upstream_node.output;

    def update_weight(self,rate):
        #根据随机梯度下降算法更新权重
        #根据式5 w41 = w41 + rate*e4*x1 w84 = w84 + rate*e8*a4
        # e8为8结点的误差项 a4位隐藏层4结点对输出值（即该连接由结点4指向结点8的输入）
        #       w51 = w51 + rate*e5*x1
        self.calc_gradient();
        # self.weight = self.weight + rate * self.downstream_node.delta * self.upstream_node.output;
        self.weight = self.weight + rate * self.gradient;

    def get_gradient(self):
        return self.gradient;

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f ' % (
         self.upstream_node.layer_index,self.upstream_node.node_index,
         self.downstream_node.layer_index,self.downstream_node.node_index,
         self.weight)
class Connections(object):
    # 提供集合操作
    def __init__(self):
        self.connections = [];

    def add_connection(self, conn):
        self.connections.append(conn)

    def dump(self):
        for conn in self.connections:
            print conn



class Network(object):
    def __init__(self, layers):
        '''
        layers:一维数组 描述第几层的结点数
        self.layers: 二维数组 描述每层结点
        '''
        self.connections = Connections();
        self.layers = [];
        layer_len = len(layers);
        node_len = 0;
        # 初始化神经网络层
        for i in range(layer_len):
            self.layers.append(Layer(i, layers[i]))
        # 初始化层与层之间的连接connections
        for i in range(layer_len - 1):
            #connections里是两层结点的笛卡儿积 元素个数为两层结点个数的乘积
            connections = [Conntection(upstream_node, downstream_node)
                           for upstream_node in self.layers[i].nodes
                           for downstream_node in self.layers[i + 1].nodes[:-1]];  # 下游结点用于反向传播计算不包括偏置项结点
            # 初始化 将生成好的连接绑定到每个结点上便于后面输出值和误差项的计算
            # 每个连接的上下游结点添加一个下上游连接 并且加入到连接集合里
            for conn in connections:
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)
                self.connections.add_connection(conn)

    def train(self, labels, data_set, iteration, rate):
        '''

        :param labels: 标签二维数组，每个样本为一个行向量，
        该行向量中最大的数所对应的列属性为对应当实际值
        例如数字识别1-〉[0.1 0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]1对应的标签向量）
        :param data_set: 二维数组 一行代表一个样本，列属性代表特征
        :param iteration: 迭代次数
        :param rate: 学习速率 即随机梯度下降时的每一步的步长
        :return:
        '''
        for i in range(iteration):
            for index in range(len(data_set)):
                self.train_one_sample(labels[index], data_set[index], rate)

    def train_one_sample(self, label, sample, rate):
        '''
        predict 正向传播使每个结点都有输出值
        calc_delta 反向传播先计算每个神经元结点的误差项
        update_weight 更新每个连接上的权重值 w = w + rate*delta*xji delta为结点j的误差项，xji为结点i传递给结点j的输入
        :param label:
        :param sample: 一个样本
        :param rate:
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self, sample):
        # 根据样本预测输出值 即得到输出层的值
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        # 返回输出层所有神经元，不包括最后一个常量神经元（记录偏置项wb）
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def calc_delta(self, label):
        # 反向计算误差项 先由layers获得输出层所有结点 从输出层从上到下的结点依次计算误差项（为一个计算函数）
        # 接着依次从后向前计算隐藏层每个结点的误差项 最后完成除输入层结点外每个结点的误差项计算
        output_nodes = self.layers[-1].nodes;
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])

        for layer in self.layers[-2::-1]:
            for node in layer.nodes[:-1]:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        # 正向更新每个结点到其下游所有结点的连接上绑定的权重
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def get_gradient(self, label, sample):
        # 获得网络在一个样本下，每个连接的梯度
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def calc_gradient(self):
        #正向更新每个结点到其下游所有结点的连接上绑定的梯度
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def dump(self):
        for layer in self.layers:
            layer.dump();

def gradient_check(net,sample_feature,sample_label):
    #定义目标函数 即所有输出结点的误差平方和Ed ,reduce(lambda x, y:x+y,-- ,0.0)求和
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda x, y: x+y,
                     map(lambda v:(v[0]-v[1])*(v[0]-v[1]),zip(vec1,vec2))
                     ,0.0)
    net.get_gradient(sample_feature,sample_label);

    for conn in net.connections.connections:
        #实际的梯度应为 conn.get_gradient()的相反数 具体看式5的推导过程
        #wji = wji - rate*梯度
        #而get_gradient()是由wji = wji + rate* e*xji得到的是e*xji
        #所以e*xji = - 梯度 即实际梯度 = - e*xji = - conn.get_gradient()
        actual_gradient =  - conn.get_gradient();
        weight = conn.weight;
        epision = 0.0001;

        #根据目标函数定义求 Ed+
        conn.weight = weight + epision
        error1 = network_error(net.predict(sample_feature),sample_label);

        # 根据目标函数定义求 Ed-
        conn.weight = weight - epision
        error2 = network_error(net.predict(sample_feature),sample_label)

        #梯度计算值
        expected_gradient = (error1 - error2) / (2 * epision)

        print 'actual_gradient: \t%f\n expected_gradient: \t%f' % (actual_gradient,expected_gradient)



class B(object):
    def __init__(self, num):
        self.num = num;


class A(object):
    def __init__(self):
        self.arrays = [];

    def add(self, b):
        self.arrays.append(b)

    def st(self):
        return self.arrays


if __name__ == '__main__':
    # a = [3, 4, 5, 6, 7, 8];
    # # [-2::-2]第二个负数代表数组倒置正数为正序绝对值为步长
    # # 第一个负数代表起始位置，负代表从后往前数-1 -2 -3...,正代表从前往后数0，1，2...,
    # # 绝对值代表位置起始数包含绝对值
    # # 第一个-2表示倒数第2个数开始（包括)第二个-2表示从后往前步长为2
    # print(a[:-5:1])
    # for i in range(1, 4 - 1):
    #     print(4 - i);
    # b = B(3);
    # a = A();
    # a.add(b);
    # b.num = 1;
    # print b.num
    # for v in a.st():
    #     print v.num
    # connections = [Conntection(upstream_node, downstream_node)
    #                for upstream_node in range(2,4)
    #                for downstream_node in range(5,9)];
    # print(len(connections))
    # b =[[1],[2]]
    # #reduce函数遍历加求和公式如下 lambda sum,x 默认第二个参数x为迭代对象
    # print reduce(lambda sum,x: sum+x[0],b,0)

    #构造网络
    net = Network([2,2,2])
    sample_feature = [0.9,0.1]
    sample_label = [0.9,0.1]
    gradient_check(net,sample_feature,sample_label)