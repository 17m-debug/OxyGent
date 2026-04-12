import asyncio
import os

from oxygent import MAS, Config, oxy, preset_tools

Config.set_agent_llm_model("default_llm")

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ChatAgent(
        name="planner_agent",
        prompt="""
The origin plan is:
{past_plan}

We have finished the following steps:
{past_steps}

Please update the plan considering the mentioned information.
Otherwise, please update the plan. The plan should only contain the steps to be executed, and do not 
include the past steps or any other information.
Please reply in JSON list format only, nothing else
""",
    ),
    preset_tools.time_tools,
    preset_tools.file_tools,
    oxy.ReActAgent(
        name="executor_agent",
        additional_prompt="You should only execute the current step, and do not execute other steps in our plan. Do not execute more than one step continuously or skip any step.",
        tools=["time_tools", "file_tools"],
    ),
    oxy.PlanAndSolveAgent(
        name="master_agent",
        is_master=True,
        planner_agent="planner_agent",
        executor_agent="executor_agent",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(
            first_query="What time is it now? Please save it into time.txt."
        )


if __name__ == "__main__":
    asyncio.run(main())
