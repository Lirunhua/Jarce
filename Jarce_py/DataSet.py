# this file includes all file used to calculate the information between attributes

# 计算两个属性之间的互信息：1.给所有数据得到一个互信息 2.给两个属性，计算两个属性间的互信息；
import math
import pickle
import Global_V

class Dataset:
    """
    Dataset:
    注意：任何数据，如果后面有set，说明是一个Dataset类，如traindata是纯数据，traindataset就是一个dataset类

    data: 原始数据，是对meta数据编码并打乱的int类型（编码：把原始的字符串（string）编码成数字（int），便于编程）
    each_attrname_array：每一行包含一个属性可能的取值（未被编码），如：

                                                                        [['vhigh', 'high', 'med', 'low'],
                                                                         ['vhigh', 'high', 'med', 'low'],
                                                                         ['2', '3', '4', '5more']](未列全)

                                                                        最后一个是类的可能取值（未被编码）
    row: 数据的行数
    col: 数据的列数
    totalcount： 数据中instance的数量(count)，也就是行数
    """
    def __init__(self, data):

        self.data = data
        # self.trainData, self.testData = self.splitdataset(0.9, 0)
        self.each_attrname_array = pickle.load(open(Global_V.TESTFILE +'_each_attrname_array.pkl', 'rb'))
        self.row = len(self.data)
        self.col = len(self.data[0])
        self.totalcount = len(self.data)
        # self.attrName_list = pickle.load(open('attrname_list.pkl','rb'))    #dict not name

    def splitdataset(self, splitratio, startline):
        # To split the train data and test data
        # startline: the line start to split, count from 0
        trainsize = int(len(self.data) * splitratio)
        trainset = []
        testdata = list(self.data)  # 注意这种拷贝方式存在的问题

        while len(trainset) < trainsize:
            if len(testdata) <= startline:
                startline = 0
            trainset.append(testdata.pop(startline))

        return [trainset, testdata]

# --------------------------------------get methods---------------------------------------- #
    def getSize(self):
        return self.row, self.col

    def getNoClass(self):
        """获得所有类的可能取值的数量，比如这个数据集里共有2个类：0,1"""
        # get the number of all the class
        return len(self.each_attrname_array[-1])   # the last element of eachattrNum_list: is "class"
    
    def getCountOfAttr_i(self, i):
        """获得某个属性i的可能取值的数量，比如属性：长度，可能取1,2,3三个属性"""
        # get Count of a specific(num) attribute
        return len(self.each_attrname_array[i])

    def getNoAttr(self):
        """获得所有属性可能取值的数量，比如数据集里共有6个属性"""
        return (len(self.getAttrName_array())-1)

    def getAttrName_array(self):
        # 在执行每个文件之前，要把这些所有的需要的pkl文件准备好
        return self.each_attrname_array

    # ----------------------------------get count---------------------------------#

    def getClassCount(self, y):
        """获取属于某个类的实例的数量"""
        # get the count of a specific class
        count = 0
        list_class = [each[-1] for each in self.data]
        for each in list_class:
            if each == y:
                count += 1
        return count

    def getXCount(self, a, v):
        """获取属性a的值为v的实例的数量"""
        # get the count of a specific attribute
        count = 0
        array_a = [each[a] for each in self.data]
        for each in array_a:
            if each == v:
                count += 1
        return count

    def getXXCount(self, a1, v1, a2, v2):
        """获得属性a1 = v1, 同时 a2 = v2的实例的数量"""
        # get the count of (a1 = v1, a2 = v2)
        array_a1 = [each[a1] for each in self.data]
        array_a2 = [each[a2] for each in self.data]
        array_a1a2 = list(zip(array_a1, array_a2))
        count = array_a1a2.count((v1, v2))
        return count

    def getXYCount(self, a, v, y):
        # get the count of a attribute a equals x, and the class equals y;
        # get the vectors of [a,y] and form a list: list_temp
        array_a = [each[a] for each in self.data]
        array_y = [each[-1]for each in self.data]
        array_ay = list(zip(array_a, array_y))
        count = array_ay.count((v, y))
        return count

    def getXXYCount(self, a1, v1, a2, v2, y):
        array_a1 = [each[a1] for each in self.data]
        array_a2 = [each[a2] for each in self.data]
        array_y = [each[-1] for each in self.data]
        array_a1a2y = list(zip(array_a1, array_a2, array_y))
        count = array_a1a2y.count((v1, v2, y))
        return count

    # ------------------------------------- get probabilities--------------------------------------- #
    def getP_yi(self, yi):
        """获得数据中：实例的类变量属于yi的概率"""
        # get the probability of P(yi)
        p_yi = self.getClassCount(yi) / self.totalcount
        return p_yi

    def getCondP_xi_ya(self, ai, vi, yi):
        """获得数据中：实例的属性值ai = vi, 并且类变量属于yi的概率"""
        if self.getXYCount(ai, vi, yi) == 0:
            return 0
        else:
            return self.getXYCount(ai, vi, yi) / self.getClassCount(yi)

    def getCondP_xi_xayb(self, ai, vi, a1, v1, yi):
        """P(xi|xa,yb)"""
        # P(A|B) = P(A&B)/P(B)
        if self.getXXYCount(ai, vi, a1, v1, yi) == 0:
            return 0
        else:
            return self.getXXYCount(ai, vi, a1, v1, yi) / self.getXYCount(a1, v1, yi)

    # ----------------------------------------get information--------------------------------------#
    def getMutualInformation(self, mi):
        """获取互信息，需要一个mi参数来装互信息"""
        # get the mutual information chart
        # mi better be a tuple
        totalCount = self.row
        for a in range(0, self.getNoAttr(), 1):
            m = 0
            for v in range(0, self.getCountOfAttr_i(a), 1):
                for y in range(0, self.getCountOfAttr_i(-1)):
                    avyCount = self.getXYCount(a, v, y)
                    if avyCount:
                        m += (avyCount /totalCount) * math.log2(avyCount/((self.getXCount(a, v)/totalCount)* self.getClassCount(y)))
            mi.append(m)
        return mi

    def getCondMutInf(self, cmi):
        """获取条件互信息，需要输入一个cmi参数来装条件互信息（可以考虑不要这个参数）"""
        for x1 in range(1, self.getNoAttr(), 1):
            for x2 in range(0, x1, 1):
                m = 0
                for v1 in range(0, self.getCountOfAttr_i(x1)):
                    for v2 in range(0, self.getCountOfAttr_i(x2)):
                        for y in range(0, self.getCountOfAttr_i(-1)):
                            x1x2y = self.getXXYCount(x1, v1, x2, v2, y)
                            if x1x2y:
                                m += (x1x2y / self.totalcount) * math.log2(self.getClassCount(y) * x1x2y / ((self.getXYCount(x1, v1, y) * self.getXYCount(x2, v2, y))))
                cmi[x1][x2] = m
                cmi[x2][x1] = m
        return cmi











