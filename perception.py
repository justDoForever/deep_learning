#coding:utf-8
class Perception(object):
   def __init__(self,input_num,activator):
      self.weights = [0.0 for _ in range(input_num)];
      self.activator = activator;
      self.bias = 0.0;


   def train(self, inputs, labels,n, learning_rate):
      #训练样本,不断使预测值与标签值相接近并得到此时的每个样本输入参数对应的权重(此处self.weights为一个权重向量)和偏置项
      for i in range(n):
         self.iteration(inputs,labels,learning_rate)

   def iteration(self,inputs,labels,learning_rate):

      #inputs是样本集合，而input是单单一个样本（向量），predict（）只能预测样本对应的标签结果，有两个输入值
      # temp = map(lambda x:learning_rate*x,map(lambda x, y:x - y,labels,y));
      # self.weights = map(lambda x, y:x+y,self.weights,map(lambda x, y:x*y,temp,inputs));
      samples = zip(inputs,labels);
      #每处理一个样本就更新一次权重和偏置项，权重的个数与样本的纬度有关，也就是样本的参数，and有两个输入参数
      #所以权重有两个，根据更新规则，权重为一个向量，偏置项为一个数，这两项同时处理
      for (input,label) in samples:
         y = self.predict(input);
         temp = learning_rate * (label - y);
         self.update(input,temp);


#
   def predict(self, input):
      return self.activator(
         reduce(
            lambda x, y:x+y,map(lambda x, y:x*y,self.weights,input),0.0
            )
          + self.bias
      )

   def update(self, input,temp):
      # 更新权重 w =w + rate*(labels-y)*xi,xi为样本输入参数,w为权重向量
      # 偏移量b = b + rate*(labels-y)
      #temp=rate*(labels-y)
      self.weights = map(lambda x, y:x+y,self.weights, map(lambda x:temp*x,input));
      self.bias = self.bias + temp;

   #重写类的输出样式
   def __str__(self):
      return "weights:%s\nbias:%f " %(self.weights,self.bias);

def f(x):
   if x > 0:
      return 1
   else:
      return 0;


def getTrainDataset():
   #and 真值表
   inputs = [[0,0],[0,1],[1,0],[1,1]];
   #and
   labels = [0,0,0,1];
   #or
   # labels = [0,1,1,1];
   return inputs,labels;


def and_perception():
   '''
   利用and真值表训练 and感知器
   :return:
   '''
   #输入参数为2个，因为and为二元函数，激活函数为f：阶越函数
   p = Perception(2,f);
   inputs,labels = getTrainDataset();
   #训练,输入数据集和标签,迭代10次,学习速率(步长)设为0.1
   p.train(inputs,labels,10,0.1);
   return p;



if __name__ == '__main__':
   a = [1 for _ in range(3)]
   b = [2,3]
   print a
   print map(lambda (x,y):x*y,zip(a,b))
   # print reduce(lambda x, y:x+y,map(lambda (x,y):x*y,zip(a,b)),0.0)
   print zip(a,b)
   # for (c,d) in zip(a,b):
   #    print c," ",d
   p = and_perception();
   print p;
   print "0 and 0 = ",p.predict([0,0]);
   print "0 and 1 = ", p.predict([0, 1]);
   print "1 and 0 = ", p.predict([1, 0]);
   print "1 and 1 = ", p.predict([1, 1]);