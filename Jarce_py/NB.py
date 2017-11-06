# tan class;
# argument: trainDataSet
import pickle
import DataSet as DS
import Global_V
from output import estimate_Output


class NB:
    def __init__(self, dataset, splitratio=0.9, startline=0):

        self.dataset = DS.Dataset(dataset)
        self.trainData, self.testData = self.dataset.splitdataset(splitratio, startline)
        if Global_V.PRINTPAR == 3:
            print(self.trainData, '\n', self.testData)
        self.trainDataSet = DS.Dataset(self.trainData)
        self.testDataSet = DS.Dataset(self.testData)
        # 如何使得self.trainDataSet, self.testDataSet也是dataset对象???

    def train(self, ratio=0.9):
        return 0

    def classify(self):
        # P(y(i)|X) = P(y(i)) *　ｐ(xr|y(i)) *　|¯| P(xi|par(xi),y(i))
        # for each_test_instance in range(0, self.testDataSet.row):    # classify for each row in testDataSet
        p_y = []  # used to save P(yi|X)
        result_y = []   # save the result in each instance of the test data
        for each_test_instance in self.testDataSet.data:
            # 对每一个data中的实例
            p_y_each_inst = []  # 这个列表是每次对一个实例分类时，属于每个类的概率list
            for eachy in range(0, self.dataset.getNoClass()):
                # 对每一个类：
                p = self.trainDataSet.getP_yi(eachy) * self.trainDataSet.getCondP_xi_ya(0, each_test_instance[0], eachy)
                # 先算P(yi)*P(xi|ya)
                # caution: it's !!!trainDataSet!!!.getP_yi
                # here we calculate the P(x0|yi)
                # 后期代码应该把每一个p_yi用pickle存起来，加快速度
                for eachx in range(1, self.dataset.getNoAttr()):  # and here we cal from x1: P(x1|x0,yi)
                    # print('each:\n', each)
                    p = p * (self.trainDataSet.getCondP_xi_ya(eachx, each_test_instance[eachx], eachy))
                    # caution: the arguments: a1, v1, a2, v2, yi should be transform to the number
                    # not the real string attributes
                p_y_each_inst.append(p)  # 属于每个类的概率
            result_y.append(p_y_each_inst.index(max(p_y_each_inst)))  # 分类结果
            p_y.append(p_y_each_inst)
        return p_y, result_y    # 任何classify函数都必须给出父节点序列(type: list)以及分类结果(type: list)

# 交叉验证
if __name__ == '__main__':
    """如果在这里运行，不进行交叉验证，用最后1/10的数据作为测试数据"""
    data_initial = pickle.load(open(Global_V.TESTFILE + '_data_list_file.pkl', 'rb'))
    data_len = len(data_initial)
    if Global_V.PRINTPAR == 3:
        print('\nThere are', data_len, 'instance in this data.\n')  # 输出数据的行数，检验数据是否载入正确
    # if the schema is TAN:
    if Global_V.SCHEME.upper() == 'NB':
        '''
        1. 载入数据
        2. 训练数据
        3. 分类
        4. 输出结果评估
           '''
        NB_data = NB(data_initial, 0.9, )
        NB_data.train()
        p_c, result = NB_data.classify()
        estimate_Output(NB_data.testData, p_c, result, 2)
        # 每个分类的过程，必须保存：1. 每一个testdata里的instance分类到每个Ci的概率p_y    2. 分类结果：result
        # output(result, TAN_car.testData,1)    # output 函数的输入为：
        # output的输出，应该要有：1.每个testdata分到每个类的概率  2.分类结果result  3.testData，(以及参数1,2,3;不同的参数，输出不同)
        # compare the prediction results with the real results
        if Global_V.PRINTPAR == 3:
            print('testset:\n', NB_data.testDataSet, '\n')
            print('p_ci= \n', p_c, '\n', 'result:\n', result)
            print('the first element is the result of prediction, second is real result!\n')

    if Global_V.SCHEME.upper() == 'KDB':
        pass
    if Global_V.SCHEME.upper() == 'AODE':
        pass
    if Global_V.SCHEME.upper() == 'NB':
        pass

