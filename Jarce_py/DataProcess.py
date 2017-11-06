import pickle
import TAN as T
import Global_V

data_initial = pickle.load(open(Global_V.TESTFILE + '_data_list_file.pkl', 'rb'))   # data_list_file.pkl 还是原始数据，并且是字符，不是数字矩阵，不是数字！！！
if Global_V.PRINTPAR == 3:
    print(len(data_initial))

# data_initial_DS = DS.Dataset(data_initial)
# trainSet, testSet = data_initial_DS.splitdataset(0.9, 90)
# if Global_V.PRINTPAR == 3:
#     print(len(trainSet), len(testSet))
#
# TrainData_DS = DS.Dataset(trainSet)
# TestData_DS = DS.Dataset(testSet)
# column, row = data_initial_DS.getSize()
# # test the methods of class Dataset
# # print('\n', , '\n')
# if Global_V.PRINTPAR == 3:
#     print('column:\n', column, '\n', 'row', row, '\n')
#     print('count of class:\n', data_initial_DS.getNoClass(), '\n')
#     print('AttrName_array:\n', data_initial_DS.getAttrName_array(), '\n')
#     print('Number of Attributes:\n', data_initial_DS.getNoAttr(), '\n')
#     print('Count of Attr 3: \n', data_initial_DS.getCountOfAttr_i(3), '\n')
#     print('Count of class 2:\n', data_initial_DS.getClassCount(2), '\n')
#     print('count of：attr(1) = 2: \n', data_initial_DS.getXCount(1, 2), '\n')
#     print('count of:　attr(1) = 2, attr(2) = 3:\n', data_initial_DS.getXXCount(1, 2, 2, 3), '\n')
#     print('count of: attr(2) = 1, ci = 2:\n', data_initial_DS.getXYCount(2, 1, 2), '\n')
#
# mi = []
# cmi = [[0]*TrainData_DS.getNoAttr() for i in range(TrainData_DS.getNoAttr())]
# graph = TrainData_DS.getConMutInf(cmi)
# if Global_V.PRINTPAR == 3:
#     print('the mutual information of traindata is:\n', TrainData_DS.getMutualInformation(mi), '\n')
#     print('the conditional mutual information of traindata is:\n', graph, '\n')

# n = len(graph)
# print(n,'\n')
# dis, pre = par.prim(graph, n)
# print(dis, '\n', pre)
# 经验证，互信息的求解数值没有任何问题；在OneNote上有图片记录
# 在设计计算预测概率的结果的时候，要记得预留得到概率的方法，这样有助于程序的拓展；
# 分类
# for each in [eachy[-1] for eachy in testSet]:
    # find the probality of each test instance
# p_eachy.append(germanTestData.getClassCount(each)/germanTestData.totalcount)
    # every class in testSet

# for each in testSet:

TAN_car = T.TAN(data_initial, 0.9, 100)
TAN_car.train(TAN_car.cmi)
if Global_V.PRINTPAR == 3:
    print('parents of each nodes(attributes):\n', TAN_car.parents, '\n')
p_c, result = TAN_car.classify()

# compare the prediction results with the real results
compare = list(zip(result, [each[-1] for each in TAN_car.testData]))
if Global_V.PRINTPAR == 3:
    print('testset:\n', TAN_car.testData, '\n')
    print('p_ci= \n', p_c, '\n', 'result:\n', result)
    print('the first element is the result of prediction, second is real result!\n')
    print('---------------------------------compare---------------------------------\n', compare)
right = 0
wrong = 0
for each in compare:
    if each[1] == each[0]:
        right += 1
    else:
        wrong += 1
loss = wrong/TAN_car.testDataSet.row


print('right:\n', right, '\n')
print('wrong:\n', wrong, '\n')
print('0-1 loss:\n', loss, '\n')





