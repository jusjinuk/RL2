import logging
from math_verify import parse, verify

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def step(state, action, extra_info):

    if "</think>" not in action and "</think>" not in state:
        env_response = {
            "next_state": None,
            "reward": 0.0,
            "score": 0.0,
            "done": False,
            "extra_info": extra_info
        }
        env_response["next_state"] = state + action + "\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
        return env_response

    if "</think>" in action:
        action = action.split("</think>")[1]

    reward = float(
        verify(
            parse(extra_info["answer"]),
            parse(action)
        )
    )
    return {
        "next_state": None,
        "reward": reward,
        "score": reward,
        "done": True,
        "extra_info": extra_info
    }
