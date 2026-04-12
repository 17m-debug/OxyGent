"""gemma_cli_demo.py"""

import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        base_url="http://localhost:11434/api/chat",
        model_name=os.getenv("DEFAULT_OLLAMA_MODEL"),
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
