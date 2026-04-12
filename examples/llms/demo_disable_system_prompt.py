import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("CHATRHINO_750B_API_KEY"),
        base_url=os.getenv("CHATRHINO_750B_BASE_URL"),
        model_name=os.getenv("CHATRHINO_750B_MODEL_NAME"),
        is_disable_system_prompt=True,  # 适配少部分不支持system prompt的模型
    ),
    oxy.ChatAgent(
        name="master_agent",
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
