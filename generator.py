import random
import requests
import json
import os
from tqdm import tqdm
import re
import networkx as nx
from typing import Tuple
from matplotlib import pyplot as plt

TOPIC_TYPES = ['Gender Discrimination', 'Climate Change', 'Racial Discrimination', 'Technology Privacy', 'Economic Inequality']
MAX_TOKENS_PERSONA = 50
MAX_TOKENS_BACKGROUND = 150

class AgentGenerator:
    def __init__(self):
        self.api_key = "sk-a109c5c9164546a7af94dc423765ed45"
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        # self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_url = "http://localhost:8111/v1/chat/completions"
        self.topics = TOPIC_TYPES
        self.agents_cache = {}

    def get_llm_response(self, prompt, max_tokens=None, expect_json=False):
        """通用 LLM 调用函数"""
        try:
            headers = {
                    # "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            # data = {
            #     "model": "deepseek-chat",
            #     "messages": [{"role": "user", "content": prompt}],
            #     "max_tokens": 5000,  # 增加token以容纳更多内容
            #     "temperature": 0.7
            # }
            data = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5000,  # 增加token以容纳更多内容
                "temperature": 0.7
            }

            success = False
            # print(prompt)
            for attempt in range(3):
                try:
                    response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
                    response.raise_for_status()
                    result = response.json()
                    # print(result['choices'][0]['message']['content'])
                    if expect_json:
                        answer = result['choices'][0]['message']['content'].strip()
                        answer = re.sub(r'^```json\s*', '', answer)
                        answer = re.sub(r'```$', '', answer).strip()
                        answer = json.loads(answer)
                    else:
                        answer = result['choices'][0]['message']['content'].strip()
                    success = True
                    break
                except Exception as e:
                    print(f"Attempt {attempt+1} error: {e}")
            if not success:
                print(f"Failed to generate agent")
            # print(answer)
            return answer

        except Exception as e:
            print(f"调用 LLM 时出错: {e}")
            return None


    def generate_demographics(self, base_info):
        prompt = f"""
    Based on the following basic agent information:
    Age: {base_info['age']}
    Gender: {base_info['gender']}

    Please generate plausible demographic details for this person living in a US coastal city.
    The race must be randomly chosen from ONLY the following list: ["White", "Black", "Asian", "Native American", "Pacific Islander"].
    The marriage_status must be randomly chosen from ONLY the following list: ["Single", "Married", "Divorced", "Widowed", "Separated"].
    Education and occupation can be any reasonable value.
    Provide your response as a single JSON object with ONLY the following keys: "race", "education", "occupation", "marriage_status".
    Example: {{"race": "White", "education": "bachelor's degree", "occupation": "teacher", "marriage_status": "Single"}}
    """
        return self.get_llm_response(prompt, expect_json=True)

    def generate_personality(self, profile_data):
        prompt = f"""
    You are a character designer. Based on the following agent profile:
    Name: {profile_data['name']}
    Age: {profile_data['age']}
    Gender: {profile_data['gender']}
    Race: {profile_data['race']}
    Education: {profile_data['education']}
    Occupation: {profile_data['occupation']}
    Marital Status: {profile_data['marriage_status']}

    Write a short, descriptive personality for this character in one short sentence in English.
    """
        return self.get_llm_response(prompt, max_tokens=MAX_TOKENS_PERSONA)

    def generate_background_story(self, profile_data):
        prompt = f"""
    Based on all the information for the following character, write a brief, plausible background story.
    Keep it concise and focused on their life journey.

    Character Profile:
    Name: {profile_data['name']}
    Age: {profile_data['age']}
    Gender: {profile_data['gender']}
    Race: {profile_data['race']}
    Education: {profile_data['education']}
    Occupation: {profile_data['occupation']}
    Marital Status: {profile_data['marriage_status']}
    Personality: {profile_data['persona']}

    Write the background story in less than 3 sentences in English.
    """
        return self.get_llm_response(prompt, max_tokens=MAX_TOKENS_BACKGROUND)

    def generate_topics(self, profile_data):
        prompt = f"""
        Based on the following agent profile:
        Name: {profile_data['name']}
        Age: {profile_data['age']}
        Gender: {profile_data['gender']}
        Race: {profile_data['race']}
        Education: {profile_data['education']}
        Occupation: {profile_data['occupation']}
        Marital Status: {profile_data['marriage_status']}
        Personality: {profile_data['persona']}
        Background: {profile_data['background_story']}

        Generate a JSON object that represents the agent's attitudes towards the following 5 topics:
        1. Gender Discrimination
        2. Climate Change
        3. Racial Discrimination
        4. Technology Privacy
        5. Economic Inequality

        Each attitude should be a concise sentence in English that reflects the agent's perspective or feelings about the topic.
        Provide your response as a single JSON object with ONLY the following keys: "Gender Discrimination", "Climate Change", "Racial Discrimination", "Technology Privacy", "Economic Inequality".
        Example:
        {{"Gender Discrimination": "Believes in equal opportunities for all genders.",
          "Climate Change": "Concerned about the impact on future generations.",
          "Racial Discrimination": "Advocates for diversity and inclusion.",
          "Technology Privacy": "Values personal data security.",
          "Economic Inequality": "Supports policies to reduce wealth gaps."}}
        """
        response = self.get_llm_response(prompt, expect_json=True)
        return response
        

    def generate_agents(self, num_agents, input_file, output_file):

        try:
            with open(input_file, 'r') as f:
                base_profiles = json.load(f)
        except FileNotFoundError:
            print(f"错误: 输入文件 {input_file} 未找到。")
            return
        except json.JSONDecodeError:
            print(f"错误: 无法解析 {input_file}。请确保它是有效的 JSON 文件。")
            return

        if num_agents > len(base_profiles):
            print(f"警告: 请求生成 {num_agents} 个 agent，但基础文件只包含 {len(base_profiles)} 个。将只生成 {len(base_profiles)} 个。")
            num_agents = len(base_profiles)

        generated_profiles = []
        print(f"开始生成 {num_agents} 个 agent 的信息...")

        def random_name():
            # 生成一个简单的英文名
            first_names = ["Alex", "Taylor", "Jordan", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Peyton", "Quinn", "Skyler", "Drew", "Reese", "Rowan", "Sawyer", "Emerson", "Finley", "Harper", "Logan", "Parker"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee"]
            return random.choice(first_names) + " " + random.choice(last_names)

        for i in tqdm(range(num_agents), desc="正在生成 Agent"):
            base_info = base_profiles[i]
            
            new_profile = {
                "id": i + 1,
                "name": random_name(),
                "age": base_info.get("age"),
                "gender": base_info.get("gender"),
            }

            demographics = self.generate_demographics(new_profile)
            if demographics:
                new_profile.update(demographics)
            else:
                print(f"未能为 agent {i+1} 生成人口统计信息，跳过。")
                continue
            # print(new_profile)
            personality = self.generate_personality(new_profile)
            new_profile["persona"] = personality if personality else "N/A"

            background = self.generate_background_story(new_profile)
            new_profile["background_story"] = background if background else "N/A"

            # 生成话题
            topics = self.generate_topics(new_profile)
            new_profile["topics"] = topics if topics else []

            generated_profiles.append(new_profile)

        # 6. 写入输出文件
        print(f"生成完成，将数据写入 {output_file}...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generated_profiles, f, indent=4, ensure_ascii=False)
            print("文件写入成功！")
        except IOError as e:
            print(f"写入文件时出错: {e}")


class EventGenerator:
    def __init__(self):
        self.api_key = "sk-a109c5c9164546a7af94dc423765ed45"
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        # self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_url = "http://localhost:8111/v1/chat/completions"
        self.topics = TOPIC_TYPES
        self.events_cache = {}

    def get_topics(self):
        return self.topics

    def get_events_for_topic(self, topic):
        if topic in self.events_cache:
            return self.events_cache[topic]

    def get_random_event(self, topic=None):
        if topic:
            events = self.get_events_for_topic(topic)
            if events:
                return random.choice(events)
        else:
            all_events = []
            for t in self.topics:
                all_events.extend(self.get_events_for_topic(t))
            if all_events:
                return random.choice(all_events)
        return None

    def load_existing_events(self, filename="event_file.json"):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                return {}
        return {}

    def add_events_to_json(self, topic, events_list, filename="event_file.json"):
        event_data = self.load_existing_events(filename)
        
        if topic not in event_data:
            event_data[topic] = []
        
        event_data[topic].extend(events_list)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, ensure_ascii=False, indent=4)
            print(f"{len(events_list)} events added to {filename} for topic {topic}")
        except Exception as e:
            print(f"Error saving to {filename}: {e}")

    def generate_and_append_events_for_topic(self, topic, n=10, filename="event_file.json"):
        
        if topic not in self.topics:
            print(f"Topic {topic} not in available topics")
            return

        events_list = []
        remaining = n

        while remaining > 0:
            num_to_generate = min(10, remaining)
            
            prompt = f"""Generate {num_to_generate} realistic news events related to "{topic}" that could serve as discussion points on social media. Each event should be a string with a headline and a brief description that no longer than 2 sentences.

Format: Return list of string, like:
[
  "string1",
  "string2",
  ...
]

Make sure the events are diverse and cover different aspects of the topic."""

            headers = {
                # "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # data = {
            #     "model": "deepseek-chat",
            #     "messages": [{"role": "user", "content": prompt}],
            #     "max_tokens": 8000,  # 增加token以容纳更多内容
            #     "temperature": 0.7
            # }
            data = {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8000,  # 增加token以容纳更多内容
                "temperature": 0.7
            }

            success = False
            for attempt in range(3):
                try:
                    response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
                    result = response.json()
                    events = json.loads(result['choices'][0]['message']['content'].strip())

                    # 验证返回的事件是否为字符串列表
                    if isinstance(events, list) and all(isinstance(event, str) for event in events):
                        events_list.extend(events)
                        success = True
                        remaining -= num_to_generate
                        break
                    else:
                        print(f"Attempt {attempt+1} failed for topic {topic}: invalid format")
                except Exception as e:
                    print(f"Attempt {attempt+1} error for topic {topic}: {e}")

            if not success:
                print(f"Failed to generate events for topic {topic} after 3 attempts")
                break

        if events_list:
            self.add_events_to_json(topic, events_list, filename)


def generate_lfr_graph(total_agents: int, num_communities: int) -> Tuple[nx.DiGraph, dict]:
    """
    Generates a directed graph with community structure and scale-free properties
    using the LFR (Lancichinetti-Fortunato-Radicchi) benchmark model.

    Args:
        total_agents (int): Total number of agents (nodes) in the network.
        num_communities (int): The number of communities to generate.

    Returns:
        Tuple[nx.DiGraph, dict]: A tuple containing the generated graph and a dictionary
                                 mapping community ID to a list of agent IDs.
    """
    if num_communities <= 0:
        raise ValueError("Number of communities must be greater than 0.")
    
    # Parameters are chosen to create a scale-free, community-structured graph.
    # avg_community_size = total_agents / num_communities
    # min_community = int(avg_community_size * 0.5) if avg_community_size > 2 else 1
    # max_community = int(avg_community_size * 1.5)
    min_community = num_communities
    max_community = num_communities

    # Basic validation for parameters
    if min_community < 1:
        min_community = 1
    if max_community >= total_agents:
        max_community = total_agents -1
    if min_community >= max_community:
        min_community = max(1, max_community - 1)

    try:
        G_undirected = nx.generators.community.LFR_benchmark_graph(
            n=total_agents,
            tau1=2.5,  # Power-law exponent for degree distribution
            tau2=1.5,  # Power-law exponent for community size distribution
            mu=0.1,    # Mixing parameter (low value -> strong communities)
            min_degree=2,
            max_degree=int(total_agents / 10),
            min_community=min_community,
            max_community=max_community,
            seed=42
        )
    except nx.exception.ExceededMaxIterations:
        print("LFR generator failed to converge. Falling back to a simpler model.")
        return generate_fallback_graph(total_agents, num_communities)

    # # Assign community_id and extract community structure
    # communities = {}
    # community_map = {}
    # next_community_id = 0
    # for node_idx, data in G_undirected.nodes(data=True):
    #     original_community_label = frozenset(data['community'])
    #     if original_community_label not in community_map:
    #         community_map[original_community_label] = next_community_id
    #         next_community_id += 1
        
    #     community_id = community_map[original_community_label]
    #     G_undirected.nodes[node_idx]['community_id'] = community_id
        
    #     if community_id not in communities:
    #         communities[community_id] = []
    #     communities[community_id].append(node_idx)

    

    # # Convert undirected to directed graph
    # G_directed = nx.DiGraph()
    # G_directed.add_nodes_from(G_undirected.nodes(data=True))
    
    # degrees = dict(G_undirected.degree())
    # for u, v in G_undirected.edges():
    #     # Edges point from lower degree to higher degree node (simulating following a "hub")
    #     if degrees[u] < degrees[v]:
    #         G_directed.add_edge(u, v)
    #     elif degrees[v] < degrees[u]:
    #         G_directed.add_edge(v, u)
    #     else:
    #         G_directed.add_edge(u, v)
    #         G_directed.add_edge(v, u)

    # # Ensure the graph is weakly connected
    # if not nx.is_weakly_connected(G_directed):
    #     largest_cc = max(nx.weakly_connected_components(G_directed), key=len)
    #     G_directed = G_directed.subgraph(largest_cc).copy()
        
    #     nodes_in_graph = set(G_directed.nodes())
    #     communities = {
    #         cid: [agent_id for agent_id in agent_list if agent_id in nodes_in_graph]
    #         for cid, agent_list in communities.items()
    #     }
    #     communities = {cid: agent_list for cid, agent_list in communities.items() if agent_list}

    # print(f"Generated graph with {G_directed.number_of_nodes()} nodes and {G_directed.number_of_edges()} edges.")
    # print(f"Number of communities: {len(communities)}")

    # return G_directed, communities
    return G_undirected, {}

def generate_fallback_graph(total_agents: int, num_communities: int) -> Tuple[nx.DiGraph, dict]:
    """A fallback generator using a relaxed caveman graph if LFR fails."""
    k = total_agents // num_communities
    if k == 0: k = 1
    G = nx.generators.community.relaxed_caveman_graph(num_communities, k, 0.1, seed=42)
    communities = {i: list(range(i * k, (i+1) * k)) for i in range(num_communities)}
    for i, comm_nodes in communities.items():
        for node in comm_nodes:
            if node in G:
                G.nodes[node]['community_id'] = i
    
    G_directed = nx.DiGraph(G)
    return G_directed, communities

if __name__ == "__main__":
    try:
        # generator = EventGenerator()
        # print("topics: ", generator.get_topics())
        # # n = 15
        # # for topic in generator.get_topics():
        # #     generator.generate_and_append_events_for_topic(topic, n=n)
        # print("Generate agents")
        # n = 100
        # script_dir = os.path.dirname(__file__)
        # input_path = os.path.join(script_dir, "base.json")
        # output_path = os.path.join(script_dir, "agent_file.json")
        # generator = AgentGenerator()
        # generator.generate_agents(n, input_path, output_path)

        g, dict = generate_lfr_graph(30, 4)
        nx.draw(g, node_size=30, arrowsize=6)
        plt.show()

    except ValueError as e:
        print(e)
