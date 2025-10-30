import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

print("input n and m:")
n, m = map(int, input().split())

G = nx.read_edgelist('twitter_combined.txt',
                     create_using=nx.DiGraph,
                     nodetype=int)

sub = G.subgraph(list(G.nodes)[:n]).copy()

# # 1. 创建一个空的有向图，用于存放新的稀疏图
# sparse_sub = nx.DiGraph()
# sparse_sub.add_nodes_from(sub.nodes())

# # 2. 计算一个生成树以保证连通性 (基于无向版本)
# # 这会形成一个连通的骨架
# undirected_sub = sub.to_undirected()
# # 找到最大的连通分量，以防原始子图不连通
# largest_cc = max(nx.connected_components(undirected_sub), key=len)
# sub_conn = sub.subgraph(largest_cc).copy()

# spanning_tree = nx.minimum_spanning_tree(sub_conn.to_undirected())
# # 将生成树的边（保留原始方向）添加到新图中
# for u, v in spanning_tree.edges():
#     if sub.has_edge(u, v):
#         sparse_sub.add_edge(u, v)
#     elif sub.has_edge(v, u):
#         sparse_sub.add_edge(v, u)

# # 3. 随机添加一些额外的边，但限制每个节点的出度
# MAX_OUT_DEGREE = 100
# ADD_EDGE_PROB = 0.2  # 添加额外边的概率

# # 遍历不在生成树中的边
# remaining_edges = set(sub.edges()) - set(sparse_sub.edges())
# for u, v in remaining_edges:
#     # 如果u的出度小于限制，并且随机成功，则添加这条边
#     if sparse_sub.out_degree(u) < MAX_OUT_DEGREE and random.random() < ADD_EDGE_PROB:
#         sparse_sub.add_edge(u, v)

# # 使用新的稀疏图进行分析
# sub = sparse_sub

# plt.savefig("twitter_subgraph.jpg")

def graph_report(G, name=''):
    print(f'====== {name} 文字体检表 ======')
    # print(nx.info(G))
    print('平均聚类 C:',      nx.average_clustering(G.to_undirected()))
    print(' reciprocity :',   nx.reciprocity(G))
    print(' 强连通大小 :',    len(max(nx.strongly_connected_components(G), key=len)))
    print(' 平均最短路径:',   nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) else '图不连通，取最大强连通分量')
    # 中心性前 10
    print(' 出度中心性 TOP10:')
    dc = nx.out_degree_centrality(G)
    for n,c in sorted(dc.items(), key=lambda x:-x[1])[:10]:
        print(f'   {n}: 中心性 {c:.3f}, 出度 {G.out_degree(n)}')

in_deg  = dict(sub.in_degree())
out_deg = dict(sub.out_degree())
df = pd.DataFrame({'in':in_deg, 'out':out_deg})
print(df.describe().round(2))
graph_report(sub, 'BA-200')

def simulate_propagation(G, m=3):
    """
    从 m 个出度最高的节点开始，模拟信息传播过程。

    规则：
    - 选择出度最高的 m 个节点作为初始激活集（如果节点数不足则全部选中）。
    - 每个激活节点在每个时间步随机向其未被激活的后继中传播 1 到 2 条边（若可用）。
    - 打印每一时间步新激活节点数与累计激活数。
    """
    if not G.nodes():
        print("图中没有节点，无法开始模拟。")
        return

    # 1. 选择出度最高的 m 个节点作为起点
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    if len(out_degrees) == 0:
        print("图中没有边或节点，无法选择起始节点。")
        return

    m = max(1, int(m))
    top_nodes = [node for node, deg in out_degrees[:m]]

    # 初始化
    activated = set(top_nodes)
    active_front = set(top_nodes)
    time_step = 0

    print("\n====== 信息传播模拟 (从出度最高的 m 个节点开始) ======")
    print(f"起始节点 (m={m}): {top_nodes}")

    # 按时间步展开传播
    while active_front:
        time_step += 1
        newly_activated = set()

        for node in active_front:
            neighbors = list(G.successors(node))
            unactivated = [n for n in neighbors if n not in activated]
            if not unactivated:
                continue
            # 每次只激活一条出边（如果有未激活的邻居）
            if unactivated:
                nb = random.choice(unactivated)
                if nb not in activated:
                    newly_activated.add(nb)

        activated.update(newly_activated)
        active_front = newly_activated

        print(f"时间步 {time_step}: 新激活 {len(newly_activated)} 个，累计激活 {len(activated)} 个")

    # 结束输出
    total = G.number_of_nodes()
    print("\n--- 模拟结束 ---")
    print(f"总时间步: {time_step}")
    print(f"最终激活: {len(activated)}/{total} 节点 ({(len(activated)/total*100) if total>0 else 0:.2f}%)")

# 从 top m 个出度最高节点开始模拟，修改下面的 m 值以改变起始节点数量

simulate_propagation(sub, m=m)

nx.draw(sub, node_size=30, arrowsize=6)
plt.show()
