# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


from typing import List, Optional
from openenv.core.env_server import Action, Observation, State


class WordGameAction(Action):
    guess: str  # The player's guessed letter

class WordGameObservation(Observation):
    # done: bool and reward: Optional[float] are already in Observation base
    masked_word: str           # e.g., "h_ll_"
    guessed_letters: List[str]
    attempts_remaining: int
    message: str               # Feedback message

class WordGameState(State):
    # episode_id: Optional[str] and step_count: int are already in State base
    target_word: str = ""
    max_attempts: int = 10