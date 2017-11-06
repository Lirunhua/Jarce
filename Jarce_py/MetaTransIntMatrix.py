# function: transfer the string data to int data
# sample:

# vhigh,vhigh,2,2,small,low,unacc
# vhigh,vhigh,2,2,small,med,unacc
# high,vhigh,2,2,small,high,unacc
# vhigh,vhigh,2,2,med,low,unacc

# after the process, will be:
# [[0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 2, 0],
#  [0, 0, 0, 0, 1, 0, 0]]

# 注意，对于范围性的属性以及缺少属性值的文件（即：连续值），这个文档还没有提供处理方式
import re
import pickle
import Global_V
import random

"""
目前只能处理完整数据和离散数据！！！
"""

# load a specific data file (we define the filename in Globle_V.py)
"""加载某个文件，这个filename定义在　Globle_V.py"""


def loaddata(filename):
    # f = open('E:\\test\\'+filename)
    f = open(r'/home/lirh/Documents/Bayes/data/test/' + filename)
    return f


file = loaddata(Global_V.TESTFILE + '.arff')  # 文件句柄

"""获得属性名称"""
attr_list = []
for each in file:
    if each[0:10] == '@attribute':  # why it's 0:10, not 0:9  ?
        pattern_attr = re.compile(r'\t\w*')
        attr_temp = re.search(pattern_attr, each)
        attr_list.append(str(attr_temp.group()))
        # successful find the attribute
# pattern_attr = re.compile('\{(.*)\}')
# car_attr = re.search(pattern_attr, car_data)
# print(car_attr.group())


attr_str_list = []
for each in attr_list:
    temp = each.strip()  # strip() 方法用来去除字符串中的空格、制表等符号；
    attr_str_list.append(temp)
# ***attr_str_list***sample***:　['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

""" transfer each element in  attr_str_list from string type to int type, we get attr_num_list"""
attr_num_list = []
ranknum = len(attr_str_list)  # 记录属性个数
for each in range(0, ranknum, 1):
    attr_num_list.append(each)

# transfer each element in  attr_str_list from string type to int type, we get attr_num_list

# *****************************attr_str_list****************************-----------sample----------:　
# ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
# *****************************attr_num_list****************************-----------sample----------:　
# [0, 1, 2, 3, 4, 5, 6]

attr_dict = dict(map(lambda x, y: [x, y], attr_num_list, attr_str_list))
# *****************************attr_dict****************************-----------sample----------:　
# {0: 'buying', 1: 'maint', 2: 'doors', 3: 'persons', 4: 'lug_boot', 5: 'safety', 6: 'class'}

# save attr_dict as attr_dict_pickle
attr_dict_pickle = open(Global_V.TESTFILE + '_attr_dict_pickle.pkl', 'wb')
pickle.dump(attr_dict, attr_dict_pickle)
attr_dict_pickle.close()
# notion: this pickle is a dict

# deal with .pmeta file
"""
::Created by arff2petal from car.arff. No other options specified.
::source relation: car

buying:vhigh,high,med,low
maint:vhigh,high,med,low
doors:2,3,4,5more
persons:2,4,more
lug_boot:small,med,big
safety:low,med,high
class:unacc,acc,good,vgood
"""
attrname_list = []
each_attrname_array = []
attrnum_dict = {}
attrnum_replace_list = [[]]
file_pmeta = loaddata(Global_V.TESTFILE + '.pmeta')  # 这里是一次性读进内存，后续如何改进？
# typical data of a instance in the pmeta file is like: (num-of-cylinders:eight,five,four,six,three,twelve,two)
for each in file_pmeta:  # 这里是一次性读取吗？　还是一行一行的读取？
    if (each[0] == ':') or (each == '\n'):  # and (each[1] == ':'):   # deal with the ':', and '\n' in the file，　
        continue
    else:
        attr_temp = each.split(':')  # split the name of attribute with the values of attribute
        # print(attr_temp);
        each_attrname_array.append(attr_temp[1].strip())
        attrname_list.append(attr_temp[0].strip())
attrnum_list_temp = []
for each in each_attrname_array:
    attrnum_list_temp.append(list(map(str, each.strip().split(','))))
each_attrname_array = attrnum_list_temp

# 处理后: each_attrname_array
""" 
[
['vhigh', 'high', 'med', 'low'], 
['vhigh', 'high', 'med', 'low'], 
['2', '3', '4', '5more'],
['2', '4', 'more'], 
['small', 'med', 'big'], 
['low', 'med', 'high'], 
['unacc', 'acc', 'good', 'vgood']
]
"""

# 把 filename.pdata 中所有的str全部转换为int数字的形式

file_pdata = loaddata(Global_V.TESTFILE + '.pdata')
data_list = []
rownum = 0  # 行数

# 这里是不是一行一行读取到内存，如果不是，需要改进？
for line in file_pdata.readlines():
    data_list.append(list(map(str, line.strip().split(','))))  # 这里的这个data_list是不是很大？怎么做？
    rownum += 1

data_num_list = []
for each_line in data_list:
    attCount = 0
    each_line_substitude = []
    for each_element in each_line:  # each_line: vhigh,vhigh,2,2,big,high,unacc
        each_line_substitude.append(each_attrname_array[attCount].index(each_element))  # 把每一行的字符型，转为num型
        attCount += 1
    data_num_list.append(each_line_substitude)  # 收集每一行
file_pdata.close()
random.shuffle(data_num_list)  # 重要！ 这里打乱了data_num_list 的顺序，以防止训练集和测试集有规律的排序（这样会严重影响结果！） &&& 会吗 17/7/6  妈的，会

# data_num_list
# [[0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 2, 0],
#  [0, 0, 0, 0, 1, 0, 0],
#  [0, 0, 0, 0, 1, 1, 0],
#  [0, 0, 0, 0, 1, 2, 0],
#  [0, 0, 0, 0, 2, 0, 0],
#  [0, 0, 0, 0, 2, 1, 0],
#  [0, 0, 0, 0, 2, 2, 0]]

# save the data_num_list as pickle  (data_pickle)
data_pickle = open(Global_V.TESTFILE + '_data_list_file.pkl', 'wb')
pickle.dump(data_num_list, data_pickle)
data_pickle.close()

each_attrname_array_pickle = open(Global_V.TESTFILE + '_each_attrname_array.pkl', 'wb')
pickle.dump(each_attrname_array, each_attrname_array_pickle)
each_attrname_array_pickle.close()

# to prove the correction of the transfer process
# we get the numbers of colomns and rows of the pdata
print('column:  ', ranknum, '\n', 'row:   ', rownum)

# data_pickle = open('data_list_file.pkl','rb')
# data_list = pickle.load(data_pickle)
# print(data_list)
# data_list = [['' for col in range(ranknum)]for row in range(rownum)]
# a = [['1','2','3'],['4','6','1'],['1','8','9']]

"""
input: Global_V.TESTFILE
output: pickle type files as follows:
---------------------------------------------each_attrname_array_pickle---------------------------------------
[
['vhigh', 'high', 'med', 'low'], 
['vhigh', 'high', 'med', 'low'], 
['2', '3', '4', '5more'],
['2', '4', 'more'], 
['small', 'med', 'big'], 
['low', 'med', 'high'], 
['unacc', 'acc', 'good', 'vgood']
]





--------------------------------------------------attr_dict_pickle--------------------------------------------
 {0: 'buying', 1: 'maint', 2: 'doors', 3: 'persons', 4: 'lug_boot', 5: 'safety', 6: 'class'}
 
 
 
 
 

---------------------------------------------------data_pickle------------------------------------------------
 # [[0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0],
#  [0, 0, 0, 0, 0, 2, 0],
#  [0, 0, 0, 0, 1, 0, 0],
#  [0, 0, 0, 0, 1, 1, 0],
#  [0, 0, 0, 0, 1, 2, 0],
#  [0, 0, 0, 0, 2, 0, 0],
#  [0, 0, 0, 0, 2, 1, 0],
#  [0, 0, 0, 0, 2, 2, 0]]
 
 
 
"""
