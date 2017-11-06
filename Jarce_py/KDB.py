import pickle
import DataSet as DS
import Global_V

class KDB:
    def __init__(self, dataset, splitratio=0.9, startline=0):
        self.dataset = DS.Dataset(dataset)
        self.trainData, self.testData = self.dataset.splitdataset(splitratio, startline)
        if Global_V.PRINTPAR == 3:
            print(self.trainData, '\n', self.testData)
        self.trainDataSet = DS.Dataset(self.trainData)
        self.testDataSet = DS.Dataset(self.testData)
        # 如何使得self.trainDataSet, self.testDataSet也是dataset对象???
        self.parents = []
        self.cmi_temp = [[0] * self.trainDataSet.getNoAttr() for i in range(self.trainDataSet.getNoAttr())]
        self.cmi = self.trainDataSet.getCondMutInf(self.cmi_temp)  # get the cmi with the trainDataSet

    def train(self):
        a = 1
    def classify(self):

        b = 2

