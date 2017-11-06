_ = float('-inf')


def prim(graph, n):
    """# 这个函数是用来求最大生成树的，如果要求最小生成树，请把graph中的元素改成负数, 起点为 0 """
    dis = [0]*n
    pre = [0]*n
    flag = [False]*n
    flag[0] = True
    k = 0
    graph = [[i * (-1) for i in each] for each in graph]    # 把原始数据元素转为负数，称为求最大生成树的prim算法
    for i in range(n):
        dis[i] = graph[k][i]
    for j in range(n-1):
        mini = -_
        for i in range(n):
            if mini > dis[i] and not flag[i]:
                mini = dis[i]
                k = i
        if k == 0:  # 不连通
            return
        flag[k] = True
        for i in range(n):
            if dis[i] > graph[k][i] and not flag[i]:
                dis[i] = graph[k][i]
                pre[i] = k

    return pre   # return dis, pre


if __name__ == '__main__':
    n = 6
    graph1 = [
             [_, 6, 3, _, _, _],
             [6, _, 2, 5, _, _],
             [3, 2, _, 3, 4, _],
             [_, 5, 3, _, 2, 3],
             [_, _, 4, 2, _, 5],
             [_, _, _, 3, 5, _],
             ]
    dis, pre = prim(graph1, n)
    print('distance:\n', dis)
    print('parents:\n', pre)

    graph2 = [
                [0, -6, -3, _, _, _],
                [-6, 0, -2, -5, _, _],
                [-3, -2, 0, -3, -4, _],
                [_, -5, -3, 0, -2, -3],
                [_, _, -4, -2, 0, -5],
                [_, _, _, -3, -5, 0],
             ]
    Adis, Bpre = prim(graph2, n)
    print(Adis, '\n', Bpre)

