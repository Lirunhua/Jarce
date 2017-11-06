# this file is to output all the algorithm
import TAN
# import KDB
# import NB
# import AODE


def estimate_Output(testdata, p_c, result, return_par=1):    # 返回结果参数默认为1，即：只返回0-1 loss
    # 这里应该进行交叉验证操作
    #
    # return_par = 1: print the 0-1 loss only;
    #              2: print the 0-1 loss, and the right counts, wrong counts.
    #              3: you determine.
    compare = list(zip(result, [each[-1] for each in testdata]))
    right = 0
    wrong = 0
    for each in compare:
        if each[1] == each[0]:
            right += 1
        else:
            wrong += 1
    temp = wrong/(right+wrong)
    loss01 = round(temp, 3)

    if return_par == 1:
        # print('The o-1 loss is: \n', loss01)
        return loss01
    if return_par == 2:
        print('right:\n', right, '\n')
        print('Wrong:\n', wrong, '\n')
        # print('The o-1 loss is: \n', loss01)
        return right, wrong, loss01
    if return_par == 3:
        print('No more output, you can add it in Output.py!!!')





