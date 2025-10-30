import argparse
import json
import os
from openai import OpenAI
from tqdm import tqdm
import random
import string
import requests

TOPIC_TYPES = ['Gender Discrimination', 'Climate Change', 'Racial Discrimination', 'Technology Privacy', 'Economic Inequality']

VLLM_BASE_URL = "http://localhost:8011/v1"
VLLM_API_KEY = "no-key-needed"
MAX_TOKENS_PERSONA = 50
MAX_TOKENS_BACKGROUND = 150
client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)


class AgentGenerator:
    def __init__(self):
        self.api_key = "sk-a109c5c9164546a7af94dc423765ed45"
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.topics = TOPIC_TYPES
        self.agents_cache = {}

    def get_llm_response(self, prompt, max_tokens=None, expect_json=False):
        """通用 LLM 调用函数"""
        try:
            headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            data = {
                "model": "deepseek-chat",
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
                except Exception as e:
                    print(f"Attempt {attempt+1} error")
            if not success:
                print(f"Failed to generate agent")
            messages = [{"role": "user", "content": prompt}]

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

        Generate a JSON object with attitudes towards the following 5 topics:
        Gender Discrimination, Climate Change, Racial Discrimination, Technology Privacy, Economic Inequality

        Each attitude should be a short sentence in English.
        Example: {{"Gender Discrimination": "...", ...}}
        """
        return self.get_llm_response(prompt, expect_json=True)

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
                "work": base_info.get("work"),
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

def main(n, input_path, output_path):
    generator = AgentGenerator()
    generator.generate_agents(n, input_path, output_path)

if __name__ == "__main__":

    n = 3
    script_dir = os.path.dirname(__file__)
    input_path = os.path.join(script_dir, "base.json")
    output_path = os.path.join(script_dir, "agent_file.json")

    main(n, input_path, output_path)
