"""Demo for using OxyGent with multiple LLMs and an agent."""

import asyncio
import os

from oxygent import MAS, Config, OxyRequest, oxy

Config.set_es_schema_shared_data(
    {
        "properties": {
            "user_pin": {"type": "keyword"},
            "user_name": {"type": "keyword"},
        }
    }
)


def process_input(oxy_request: OxyRequest) -> OxyRequest:
    oxy_request.set_shared_data("user_pin", "123456")
    oxy_request.set_shared_data("user_name", "oxy")
    return oxy_request


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ReActAgent(
        name="master_agent",
        llm_model="default_llm",
        func_process_input=process_input,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
