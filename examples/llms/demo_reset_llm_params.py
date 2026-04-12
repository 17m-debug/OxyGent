import asyncio
import os

from oxygent import MAS, Config, oxy

Config.set_llm_config({})  # 重置LLM默认参数，以适配GPT-5模型


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"thinking": False, "stream": False},
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.call(
            callee="default_llm",
            arguments={"messages": [{"role": "user", "content": "hello"}]},
        )


if __name__ == "__main__":
    asyncio.run(main())
