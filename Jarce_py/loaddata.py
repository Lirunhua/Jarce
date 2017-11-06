# from numpy import range
import random
import DataSet as DS


# def loaddata(filename):
#     """load file, return a file - pointer: f"""
#     f = open('E:\\test\\' + filename)
#     return f














# 以下代码用于测试时，提供初始数据集
if __name__ == '__main__':
    data = [[1, 3, 34], [2, 87, 2], [3, 456], [4, 234, 42], [5, 324], [6, 562], [7, 2, 4], [8, 1, 0], [9, 1, 78],
            [10, 12, 56]]
    trainData, testData = DS.splitdataset(data, 0.9, 0)
    print(trainData, '\n', testData)
