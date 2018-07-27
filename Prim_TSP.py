'''
TSP—近似算法
1、选择G的任意一个顶点
2、Prim算法找出找出最小生成树T
3、前序遍历树T得到的顶点表L
4、将根节点添加到L的末尾，按表L中顶点的次序组成哈密顿回路H
'''
import numpy as np
import matplotlib.pyplot as plt
# 代价函数（具有三角不等式性质）
def price_cn(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
# 从去过的点中，找到连接到未去过的点的边里，最小的代价边(贪心算法)
def find_min_edge(visited_ids, no_visited_ids):
    min_weight, min_from, min_to = np.inf, np.inf, np.inf
    for from_index in visited_ids:
        for to_index, weight in enumerate(G[from_index]):
            if from_index != to_index and weight < min_weight and to_index in no_visited_ids:
                min_to = to_index
                min_from = from_index
                min_weight = G[min_from][min_to]
    return (min_from, min_to), min_weight
# 维护未走过的点的集合
def contain_no_visited_ids(G, visited_ids):
    no_visited_ids = []  # 还没有走过的点的索引集合
    [no_visited_ids.append(idx) for idx, _ in enumerate(G) if idx not in visited_ids]
    return no_visited_ids
# 生成最小生成树T
def prim(G, root_index=0):
    visited_ids = [root_index]  # 初始化去过的点的集合
    T_path = []
    while len(visited_ids) != G.shape[0]:
        no_visited_ids = contain_no_visited_ids(G, visited_ids)  # 维护未去过的点的集合
        (min_from, min_to), min_weight = find_min_edge(visited_ids, no_visited_ids)
        visited_ids.append(min_to)  # 维护去过的点的集合
        T_path.append((min_from, min_to))
    T = np.full_like(G, np.inf)  # 最小生成树的矩阵形式，n-1条边组成
    for (from_, to_) in T_path:
        T[from_][to_] = G[from_][to_]
        T[to_][from_] = G[to_][from_]
    return T, T_path
# 先序遍历图(最小生成树)的路径，得到顶点列表L
def preorder_tree_walk(T, root_index=0):
    is_visited = [False] * T.shape[0]
    stack = [root_index]
    T_walk = []
    while len(stack) != 0:
        node = stack.pop()
        T_walk.append(node)
        is_visited[node] = True
        nodes = np.where(T[node] != np.inf)[0]
        if len(nodes) > 0:
            [stack.append(node) for node in reversed(nodes) if is_visited[node] is False]
    return T_walk
# 生成哈密尔顿回路H
def create_H(G, L):
    H = np.full_like(G, np.inf)
    H_path = []
    for i, from_node in enumerate(L[0:-1]):
        to_node = L[i + 1]
        H[from_node][to_node] = G[from_node][to_node]
        H[to_node][from_node] = G[to_node][from_node]
        H_path.append((from_node, to_node))
    return H, H_path
# 可视化画出哈密顿回路
def draw_H(citys, H_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    for (from_, to_) in H_path:
        p1 = plt.Circle(citys[from_], 0.2, color='red')
        p2 = plt.Circle(citys[to_], 0.2, color='red')
        ax.add_patch(p1)
        ax.add_patch(p2)
        ax.plot((citys[from_][0], citys[to_][0]), (citys[from_][1], citys[to_][1]), color='red')
        ax.annotate(s=chr(97 + to_), xy=citys[to_], xytext=(-8, -4), textcoords='offset points', fontsize=20)
    ax.axis('equal')
    ax.grid()
    plt.show()
if __name__ == '__main__':
    citys = [(2, 6), (2, 4), (1, 3), (4, 6), (5, 5), (4, 4), (6, 4), (3, 2)]  # 城市坐标
    G = []  # 完全无向图
    for i, curr_point in enumerate(citys):
        line = []
        for j, other_point in enumerate(citys):
            line.append(price_cn(curr_point, other_point)) if i != j else line.append(np.inf)
        G.append(line)
    G = np.array(G)
    # 1、选择G的任意一个顶点
    root_index = 0
    # 2、Prim算法找出找出最小生成树T
    T, T_path = prim(G, root_index=root_index)
    # 3、前序遍历树T得到的顶点表L
    L = preorder_tree_walk(T, root_index=root_index)
    # 4、将根节点添加到L的末尾，按表L中顶点的次序组成哈密顿回路H
    L.append(root_index)
    H, H_path = create_H(G, L)
    print('最小生成树的路径为：{}'.format(T_path))
    [print(chr(97 + v), end=',' if i < len(L) - 1 else '\n') for i, v in enumerate(L)]
    print('哈密顿回路的路径为：{}'.format(H_path))
    print('哈密顿回路产生的代价为：{}'.format(sum(G[from_][to_] for (from_, to_) in H_path)))
    # draw_H(citys, H_path)
