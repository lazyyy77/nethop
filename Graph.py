import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

class Graph:
    def __init__(self, vertex):
        self.v = vertex
        self.g = nx.read_edgelist('twitter_combined.txt',
                                 create_using=nx.DiGraph,
                                 nodetype=int)
        self.graph = self.g.subgraph(list(self.g.nodes)[:self.v]).copy()
        # 重新为子图的节点编号，从 0 开始
        # 使用排序后的节点列表以保证结果可复现；如果你希望按照原始迭代顺序编号，
        # 可以把 sorted(self.graph.nodes()) 换成 list(self.graph.nodes())
        old_nodes = sorted(self.graph.nodes())
        self.node_mapping = {old: new for new, old in enumerate(old_nodes)}
        # 采用就地重命名以避免复制整个图（如果你想保留原图，请设置 copy=True）
        nx.relabel_nodes(self.graph, self.node_mapping, copy=False)
        self.e = self.graph.number_of_edges()
        # 为每条边随机分配一个权重，范围 1-10，保存在边属性 'weight' 中
        for u, v in self.graph.edges():
            self.graph[u][v]['weight'] = random.randint(1, 10)
        nx.draw(self.graph, with_labels=True)
        # plt.show()


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

if __name__ == "__main__":
    n = int(input())
    print(f"Creating graph with {n} nodes...")
    G = Graph(vertex=n).graph
    simulate_propagation(G, m=10)