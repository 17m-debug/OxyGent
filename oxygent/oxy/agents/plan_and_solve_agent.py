import json
import logging

from pydantic import Field

from ...schemas import Memory, Message, OxyRequest, OxyResponse, OxyState
from .local_agent import LocalAgent

logger = logging.getLogger(__name__)


class PlanAndSolveAgent(LocalAgent):
    max_replan_rounds: int = Field(30, description="Maximum retries for operations.")
    planner_agent: str = Field("planner_agent", description="planner agent name")
    executor_agent: str = Field("executor_agent", description="executor agent name")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_permitted_tools([self.planner_agent, self.executor_agent])

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        async def answer(past_steps_str, short_memory, original_query):
            temp_memory = Memory()
            temp_memory.add_message(
                Message.system_message(
                    f"You are an expert in task summarization. The historical execution steps are: \n{past_steps_str}.\nBased on the historical execution records and the user’s overall task, please provide the user with a final response."
                )
            )
            temp_memory.add_messages(Message.dict_list_to_messages(short_memory))
            temp_memory.add_message(
                Message.user_message(f"The current overall task is: {original_query}")
            )
            final_oxy_response = await oxy_request.call(
                callee=self.llm_model,
                arguments={"messages": temp_memory.to_dict_list()},
            )
            return OxyResponse(
                state=OxyState.COMPLETED,
                output=final_oxy_response.output,
            )

        original_query = oxy_request.get_query()
        short_memory = oxy_request.get_short_memory()
        past_plan = ""
        past_steps = []

        for current_round in range(self.max_replan_rounds + 1):
            planner_oxy_response = await oxy_request.call(
                callee=self.planner_agent,
                arguments={
                    "query": original_query,
                    "past_plan": past_plan,
                    "past_steps": "\n".join(past_steps),
                },
            )
            past_plan = planner_oxy_response.output
            plans = json.loads(past_plan)

            for current_step, current_task in enumerate(plans):
                print(current_step, current_task)
                executor_oxy_response = await oxy_request.call(
                    callee=self.executor_agent,
                    arguments={
                        "query": f"The current step to execute is: {current_task}"
                    },
                    is_async_storage=False,
                )
                past_steps.append(
                    f"task: {current_task}, execute task result: {executor_oxy_response.output}"
                )
                past_steps_str = "\n".join(past_steps)
                if current_step == len(plans) - 1:
                    return await answer(past_steps_str, short_memory, original_query)
                else:
                    temp_memory = Memory()
                    temp_memory.add_message(
                        Message.system_message(
                            f"You are an expert in supervising task execution. The historical execution steps are: \n{past_steps_str}.\nThe next step is: {plans[current_step + 1]}. \nIf the next step is reasonable, please reply “continue”; if it is unreasonable, please reply “replan”; if the task is already completed, please reply “complete”. Do not provide any other content."
                        )
                    )
                    temp_memory.add_messages(
                        Message.dict_list_to_messages(short_memory)
                    )
                    temp_memory.add_message(
                        Message.user_message(f"The overall task is: {original_query}")
                    )
                    ctrl_oxy_response = await oxy_request.call(
                        callee=self.llm_model,
                        arguments={"messages": temp_memory.to_dict_list()},
                    )
                if ctrl_oxy_response.output.startswith("complete"):
                    return await answer(past_steps_str, short_memory, original_query)
                elif ctrl_oxy_response.output.startswith("replan"):
                    break
                else:
                    pass

        return await answer(past_steps_str, short_memory, original_query)
