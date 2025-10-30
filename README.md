# nethop
net-hop situation for PF-Engine

## 项目结构

```
nethop/
├── __init__.py          # 包初始化文件
├── constants.py         # 系统常量定义
├── event.py            # Event类 - 事件管理
├── message.py          # Message类 - 消息管理
├── agent.py            # Agent类 - 智能体实现
├── environment.py      # SocialEnvironment类 - 社交环境
├── simulation.py       # SimulationEngine类 - 模拟引擎
├── run.py              # 独立运行脚本
├── main.py             # 主入口点
├── requirements.txt    # 依赖包列表
└── README.md           # 项目说明
```

## 依赖包

fastapi
uvicorn
pydantic
httpx
sqlalchemy
psycopg2-binary
pymysql
requests
networkx


生成：
1. 话题类型
2. 每种话题的事件储备
3. agent信息

Event：
    topic type
    timestamp
    content
    authenticity
    relevance
    jump

Message：
    event
    timestamp



forward信息
generate new 信息 profile emotion attitude

agent：
    emotion prediction and update based on present， profile， history， message now 衰减系数
    attitude on specific topic， same
    popular：lots of out； regular： equal in out； low：terminal
    memory pool
1 Education Practitioner
2 Administrative Manager / Officer
3 Unemployed / Student
4 Engineer
5 Labor Technician / Worker
6 Logistics Practitioner
7 Medical Personnel
8 Financial Practitioner
9 Media Personnel
10 Entertainment and Arts Practitioner

Event类
存在dict内，序号索引

message 类
传播次数 时间 相关性 -》 加权计算得到importance
    a cosine similarity between a user’s fundamental attributes and the content of the message
    authenticity：生产可信度，传播源，

请以 Python 语言为基础，设计并实现一个社交网络模拟系统。
1. 总体目标与系统架构
系统名称： Social-network Simulation System

2. 核心模块与功能要求
2.1. 环境模块
实现一个 SocialEnvironment 类，用于管理网络、用户和信息流。

网络结构： 创建一个图结构，节点代表用户，边代表社交关系。

事件接口：生成大量和议题相关的事件，并将事件发送给某些活跃的agent，让他们就此展开互动。



消息（Message）： 消息应包含 messageID、senderID、content（内容）和 type（帖子/评论/转发）。

信息流： 维护一个全局的 message_pool 或 feed，用于存放当前时间步内智能体可以感知的新消息。

2.2. agent模块

agent有如下信息：

emotion，情感状态，形容词表达

attitude：对特定话题的看法：

memory：

emotion：当前情感状态（例如，枚举类型：'Calm', 'Moderate', 'Intense'，或用数值表示）。

attitude：对特定议题的倾向性（例如，'Positive', 'Negative', 'Neutral'，或用数值表示）。

memory：智能体的记忆池，存储最具影响力的历史消息和交互记录。

核心方法：

_llm_stub_reasoning(prompt, context)：模拟LLM功能。 接收详细的提示（包含智能体状态和新消息），返回智能体更新后的状态和拟采取的行动。

perceive(message_feed)：感知环境中的新消息，过滤出与其相关或来自其社交连接的消息。

update_state(message)：将感知到的消息和当前状态输入给 _llm_stub_reasoning，更新**emotion、attitude和memory**。

act()：根据更新后的状态，生成一个交互行为（Interaction Behavior）。

2.3. 行为（Behavior Generation）模块
LLMAgent.act() 必须能够生成并输出以下四种交互行为，并将其作为新消息或网络更新反馈给环境：

like（点赞）

forward（转发/分享）

comment（评论，需生成一段文本）

generate_new_content（生成新帖子/原创内容，需生成一段文本）

2.4. 模拟引擎（Simulation Engine）模块
实现一个 SimulationEngine 类，包含主循环。

主循环（Iteration）： 迭代进行 T 个时间步。

迭代步骤： 在每个时间步 t：

环境感知： 所有智能体感知 SocialEnvironment 中的新消息。

状态更新： 每个智能体调用 update_state() 更新其内部状态。

行为生成： 每个智能体调用 act() 生成交互行为。

环境反馈： 将新生成的行为（新消息、点赞/转发记录）反馈到 SocialEnvironment 中，为下一时间步做准备。

数据记录： 记录每个时间步的群体层面现象数据，包括：

信息传播率和路径。

人群的平均态度和情感分布的变化。

## 安装和使用

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python run.py
```

这将运行一个包含5个智能体的社交网络模拟，执行5个时间步的模拟。

### 作为包使用

```python
from nethop import SocialEnvironment, Agent, SimulationEngine

# 创建环境
env = SocialEnvironment()

# 添加智能体
agent = Agent("agent_1", "Teacher")
env.add_agent(agent)

# 运行模拟
engine = SimulationEngine(env, num_steps=10)
engine.run_simulation()
```

## 系统特性

- **模块化设计**：每个类分离到独立文件，便于维护和扩展
- **网络建模**：使用NetworkX实现社交关系图
- **行为模拟**：支持点赞、转发、评论和原创内容生成
- **状态管理**：智能体具有情感、态度和记忆状态
- **事件驱动**：系统可以生成各种话题的事件驱动交互