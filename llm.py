import aiohttp
import asyncio
from typing import List, Dict, Any
import httpx
from typing import Union, List, Optional
from openai import AsyncOpenAI

class OpenAILLM:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 256,
        timeout: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            ),
        )

    # async def acomplete(self, prompt: str, agent_id: str) -> str:
    #     url = f"{self.base_url}/chat/completions"
    #     # lora_name = f"lora_{agent_id}" if agent_id is not None else "lora0"
    #     lora_name = "lora0"
    #     payload = {
    #         "model": self.model,
    #         "messages": [{"role": "user", "content": prompt}],
    #         "temperature": self.temperature,
    #         "max_tokens": self.max_tokens,
    #         "extra_body": {"agent_id": agent_id, "lora_path": lora_name},
    #     }
    #     async with aiohttp.ClientSession(timeout=self.timeout) as session:
    #         async with session.post(url, json=payload) as resp:
    #             if resp.status != 200:
    #                 text = await resp.text()
    #                 raise RuntimeError(f"LLM server error {resp.status}: {text}")
    #             data = await resp.json()
    #             # 兼容 OpenAI 返回格式
    #             return data["choices"][0]["message"]["content"].strip()

    async def acomplete(
            self,
            prompt: str,
            agent_id: str,
            temperature: float = 0,
            top_p: Optional[float] = 1.0,
            frequency_penalty: Optional[float] = 0.0,
            presence_penalty: Optional[float] = 0.0,
            timeout: int = 300,
            retries: int = 10,
        ):

            client = AsyncOpenAI(
                api_key="test",
                timeout=timeout,
                base_url=self.base_url,
                http_client=self._http_client,
            )
            message = [{"role": "user", "content": prompt}]
            
            if agent_id is None:
                agent_id = "-1"  
            if int(agent_id) >= 0:
                lora_name = f"lora{agent_id}"
            else:
                lora_name = "lora0"
            
            exb = {"agent_id": agent_id, "lora_path": lora_name}
            # exb = {"agent_id": agent_id, "lora_path": "lora0"}
            # exb = {"agent_id": agent_id}

            for attempt in range(retries):
                response = None
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=message,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stream=False,
                        timeout=timeout,
                        extra_body=exb,
                        seed=42,
                    )
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("No content in response")
                    return content
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
            raise RuntimeError("Failed to get response from LLM")

if __name__ == "__main__":
    print("Testing OpenAILLM...")
    async def test():
        llm = OpenAILLM()
        prompt = "Hello, how are you?"
        response = await llm.acomplete(prompt, agent_id="-1")
        print("LLM Response:", response)

    asyncio.run(test())
