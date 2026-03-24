
import random

from server.my_env_environment import WordGameEnvironment
from models import WordGameAction, WordGameObservation, WordGameState

WORDS = ["python", "neural", "tensor", "matrix", "vector",
         "kernel", "lambda", "signal", "binary", "cipher"]

class RandomPolicy:
    """Pure random — baseline."""
    

    def select_action(self) -> int:
        return random.choice(WORDS)

def run_episode(env, policy, verbose=False):
    """Play one episode. Returns 1 if caught, 0 if missed."""
    result = env.reset()
    step = 0

    while not result.done:
        action_id = policy.select_action()
        result = env.step(WordGameAction(guess=action_id))
        step += 1

    caught = 1 if result.reward and result.reward > 0 else 0
    if verbose:
        status = 'Caught!' if caught else 'Missed'
        print(f'  Result: {status} (reward={result.reward})')
    return caught

env = WordGameEnvironment()
policy = RandomPolicy()

episodes = 100
total_caught = 0
for ep in range(episodes):
    if ep % 10 == 0:
        print(f'Episode {ep+1}/{episodes}')
    total_caught += run_episode(env, policy)

print(f'Caught {total_caught}/{episodes} episodes ({total_caught/episodes:.2%})')
