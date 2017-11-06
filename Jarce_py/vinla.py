# 可以通用的过程：
# 1. 交叉验证
# 2. 求 0-1 loss
import pickle
import Global_V
from TAN import TAN
from NB import NB
from output import estimate_Output


data_initial = pickle.load(open(Global_V.TESTFILE + '_data_list_file.pkl', 'rb'))
data_len = len(data_initial)
if Global_V.PRINTPAR == 3:
    print('\nThere are', data_len, 'instance in this data.\n')  # 输出数据的行数，检验数据是否载入正确
loss01 = []  # 保存每次交叉验证的01loss

# if the schema is TAN:
if Global_V.SCHEME.upper() == 'TAN':  # 统一转化为大写
    '''
    1. 载入数据
    2. 训练数据
    3. 分类
    4. 输出结果评估
    '''
    result = []
    fold_count = 0
    for CrossTest_count in range(0, data_len, data_len//10):   # 交叉验证
        p_c_fold = []
        loss01_fold = 0
        TAN_data = TAN(data_initial, 0.9, CrossTest_count)
        TAN_data.train(TAN_data.cmi)
        p_c_fold, result_fold = TAN_data.classify()
        result.append(result_fold)
        # print('\n第', fold_count, '次交叉结果：\n')
        loss01_fold = estimate_Output(TAN_data.testData, p_c_fold, result_fold, 1)
        print('\n第', fold_count, '次交叉......')
        loss01.append(loss01_fold)
        fold_count += 1
        # 每个分类的过程，必须保存：1. 每一个testdata里的instance分类到每个Ci的概率p_y    2. 分类结果：result
        # output(result, TAN_car.testData,1)    # output 函数的输入为：
        # output的输出，应该要有：1.每个testdata分到每个类的概率  2.分类结果result  3.testData，(以及参数1,2,3;不同的参数，输出不同)
        # compare the prediction results with the real results
        if Global_V.PRINTPAR == 3:
            print('testset:\n', TAN_data.testData, '\n')
            print('p_ci= \n', p_c_fold, '\n', 'result:\n', result)
            print('the first element is the result of prediction, second is real result!\n')
        if Global_V.PRINTPAR == 3:
            print('parents of each nodes(attributes):\n', TAN_data.parents, '\n')

    print('\n所有平均交叉结果:\n', round(sum(loss01)/len(loss01), 3))

if Global_V.SCHEME.upper() == 'KDB':
    pass
if Global_V.SCHEME.upper() == 'AODE':
    pass
if Global_V.SCHEME.upper() == 'NB':

    result = []
    fold_count = 0
    for CrossTest_count in range(0, data_len, data_len // 10):  # 交叉验证
        p_c_fold = []
        loss01_fold = 0
        NB_data = NB(data_initial, 0.9, CrossTest_count)
        NB_data.train()
        p_c_fold, result_fold = NB_data.classify()
        result.append(result_fold)
        # print('\n第', fold_count, '次交叉结果：\n')
        print('\n第', fold_count, '次交叉')
        loss01_fold = estimate_Output(NB_data.testData, p_c_fold, result_fold, 1)
        loss01.append(loss01_fold)
        fold_count += 1
        # 每个分类的过程，必须保存：1. 每一个testdata里的instance分类到每个Ci的概率p_y    2. 分类结果：result
        # output(result, TAN_car.testData,1)    # output 函数的输入为：
        # output的输出，应该要有：1.每个testdata分到每个类的概率  2.分类结果result  3.testData，(以及参数1,2,3;不同的参数，输出不同)
        # compare the prediction results with the real results
        if Global_V.PRINTPAR == 3:
            print('testset:\n', NB_data.testData, '\n')
            print('p_ci= \n', p_c_fold, '\n', 'result:\n', result)
            print('the first element is the result of prediction, second is real result!\n')
        if Global_V.PRINTPAR == 3:
            print('parents of each nodes(attributes):\n', NB_data.parents, '\n')

    print('\n所有平均交叉结果:\n', round(sum(loss01) / len(loss01), 3))



    #
    # NB_data = NB(data_initial)
    # NB_data.train()
    # p_c, result = NB_data.classify()
    # estimate_Output(NB_data.testData, p_c, result, 2)
    # # 每个分类的过程，必须保存：1. 每一个testdata里的instance分类到每个Ci的概率p_y    2. 分类结果：result
    # # output(result, TAN_car.testData,1)    # output 函数的输入为：
    # # output的输出，应该要有：1.每个testdata分到每个类的概率  2.分类结果result  3.testData，(以及参数1,2,3;不同的参数，输出不同)
    # # compare the prediction results with the real results
    # if Global_V.PRINTPAR == 3:
    #     print('testset:\n', NB_data.testDataSet, '\n')
    #     print('p_ci= \n', p_c, '\n', 'result:\n', result)
    #     print('the first element is the result of prediction, second is real result!\n')
