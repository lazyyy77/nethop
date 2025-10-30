import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
from typing import Optional, Dict, List


class Graph:
    def __init__(self, m: int, n: int, seed: Optional[int] = None):
        """
        手工构造 m 条长度为 n 的链状有向图。
        节点编号 0..m*n-1，第 i 条链的节点为 i*n .. i*n+(n-1)。
        每条边随机赋权 1-10（可复现）。
        同时在节点属性中存下后续第 1、2、3 个节点编号（next1, next2, next3）。
        """
        self.m = m
        self.n = n
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.graph = nx.DiGraph()
        total_nodes = m * n
        self.graph.add_nodes_from(range(total_nodes))

        for i in range(m):
            base = i * n
            for j in range(n - 1):
                u, v = base + j, base + j + 1
                self.graph.add_edge(u, v, weight=random.randint(1, 10))

        # 存后续 1、2、3 个节点编号
        for node in self.graph.nodes:
            succ = list(self.graph.successors(node))
            attrs = {
                'next1': succ[0] if len(succ) >= 1 else None,
                'next2': succ[1] if len(succ) >= 2 else None,
                'next3': succ[2] if len(succ) >= 3 else None,
            }
            nx.set_node_attributes(self.graph, {node: attrs})

        self.v = self.graph.number_of_nodes()
        self.e = self.graph.number_of_edges()

        nx.draw(self.graph, with_labels=True)
        plt.savefig("graph.jpg")

    # —— 下面方法与旧代码完全一致，直接复用 —— #
    def compute_hop_distances(self) -> Dict[int, Dict[int, int]]:
        return {src: dict(lens) for src, lens in nx.all_pairs_shortest_path_length(self.graph)}

    # —— 新增接口：返回 m 条链的头节点编号 —— #
    def get_chain_heads(self) -> List[int]:
        return [i * self.n for i in range(self.m)]

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
    m = int(input("请输入链的数量 m: "))
    n = int(input("请输入每条链的长度 n: "))
    print(f"Creating graph with {m} chains {n} lengths")
    gobj = Graph(m=m, n=n)
    print(gobj.get_chain_heads())
