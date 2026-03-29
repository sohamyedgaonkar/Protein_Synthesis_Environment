"""Baseline inference script for the protein folding OpenEnv environment.

MANDATORY
- Before running, define the following environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- This script is named `inference.py` and is placed in the project root.
- All LLM calls are made through the OpenAI client.
"""

from __future__ import annotations

import copy
import json
import os
import re
import textwrap
from typing import Any

from openai import OpenAI

try:
    from models import ProteinAction, ProteinObservation
    from server.my_env_environment import ProteinFoldingEnvironment
    from test import build_action_candidates, format_action
except ImportError:
    from my_env.models import ProteinAction, ProteinObservation
    from my_env.server.my_env_environment import ProteinFoldingEnvironment
    from my_env.test import build_action_candidates, format_action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TASK_ID = os.getenv("TASK_ID", "task_1")
EPISODE_SEED = int(os.getenv("EPISODE_SEED", "7"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "250"))
SHORTLIST_SIZE = int(os.getenv("SHORTLIST_SIZE", "8"))

ACTION_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a simplified protein folding environment.
    Your job is to choose exactly one structural action that improves the protein conformation.

    Return exactly one JSON object with this schema:
    {
      "action_type": "rotate_phi | rotate_psi | pivot_rotation | segment_flip | crankshaft_move | end_move_forward | end_move_backward",
      "residue_index": int or null,
      "segment_start": int or null,
      "segment_end": int or null,
      "angle_delta": number or null
    }

    Rules:
    - Use only one of the candidate actions provided in the user message.
    - Return valid JSON only.
    - Do not add markdown fences.
    - Do not add explanations.
    """
).strip()


def summarize_observation(observation: ProteinObservation) -> str:
    """Create a readable environment summary for the language model."""
    score_components = observation.metadata.get("score_components", {})
    return textwrap.dedent(
        f"""
        energy: {observation.energy:.3f}
        step_count: {observation.step_count}
        hydrophobic_contacts: {observation.hydrophobic_contacts}
        collisions: {observation.collisions}
        normalized_score: {float(observation.metadata.get('score', 0.0)):.3f}
        energy_reduction_ratio: {float(score_components.get('energy_reduction_ratio', 0.0)):.3f}
        hydrophobic_contact_ratio: {float(score_components.get('hydrophobic_contact_ratio', 0.0)):.3f}
        stability_score: {float(score_components.get('stability_score', 0.0)):.3f}
        first_5_torsions: {observation.torsion_angles[:5]}
        """
    ).strip()


def estimate_action_quality(observation: ProteinObservation) -> float:
    """Rank candidate actions using immediate environment feedback."""
    return (
        float(observation.metadata.get("score", 0.0)) * 100.0
        + float(observation.reward or 0.0)
        - observation.energy * 0.5
        - observation.collisions * 5.0
        + observation.hydrophobic_contacts * 2.0
    )


def shortlist_candidates(
    env: ProteinFoldingEnvironment,
    candidates: list[ProteinAction],
    shortlist_size: int,
) -> list[tuple[ProteinAction, ProteinObservation, float]]:
    """Evaluate all legal actions once and keep the strongest immediate moves."""
    ranked: list[tuple[ProteinAction, ProteinObservation, float]] = []
    for action in candidates:
        env_copy = copy.deepcopy(env)
        observation = env_copy.step(action)
        ranked.append((action, observation, estimate_action_quality(observation)))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[: max(1, shortlist_size)]


def build_user_prompt(
    observation: ProteinObservation,
    candidates: list[tuple[ProteinAction, ProteinObservation, float]],
    history: list[str],
) -> str:
    """Build the user prompt sent to the model."""
    if history:
        history_text = "\n".join(history[-5:])
    else:
        history_text = "None"

    candidate_lines = []
    for index, (action, next_obs, quality) in enumerate(candidates, start=1):
        candidate_lines.append(
            textwrap.dedent(
                f"""
                {index}. {format_action(action)}
                   estimated_next_energy: {next_obs.energy:.3f}
                   estimated_reward: {float(next_obs.reward or 0.0):.3f}
                   estimated_score: {float(next_obs.metadata.get('score', 0.0)):.3f}
                   estimated_collisions: {next_obs.collisions}
                   estimated_contacts: {next_obs.hydrophobic_contacts}
                   heuristic_quality: {quality:.3f}
                """
            ).strip()
        )

    return textwrap.dedent(
        f"""
        Task: {TASK_ID}
        Objective: lower energy, reduce collisions, and improve hydrophobic packing.

        Current environment summary:
        {summarize_observation(observation)}

        Recent action history:
        {history_text}

        Candidate actions:
        {chr(10).join(candidate_lines)}

        Choose exactly one candidate and return only the JSON object for that action.
        """
    ).strip()


def action_to_payload(action: ProteinAction) -> dict[str, Any]:
    """Convert an action model to a plain JSON dictionary."""
    return {
        "action_type": action.action_type,
        "residue_index": action.residue_index,
        "segment_start": action.segment_start,
        "segment_end": action.segment_end,
        "angle_delta": action.angle_delta,
    }


def parse_action_response(
    response_text: str,
    candidate_actions: list[ProteinAction],
) -> ProteinAction:
    """Parse the model response and fall back safely if needed."""
    if not response_text:
        return candidate_actions[0]

    json_match = ACTION_JSON_RE.search(response_text)
    if not json_match:
        return candidate_actions[0]

    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return candidate_actions[0]

    for action in candidate_actions:
        if action_to_payload(action) == payload:
            return action

    return candidate_actions[0]


def ensure_required_env() -> None:
    """Check the required model configuration."""
    missing = [
        name
        for name, value in (
            ("API_BASE_URL", API_BASE_URL),
            ("MODEL_NAME", MODEL_NAME),
            ("HF_TOKEN", API_KEY),
        )
        if not value
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_text}")


def main() -> None:
    """Run one inference episode with LLM-selected structural actions."""
    ensure_required_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = ProteinFoldingEnvironment()
    observation = env.reset(seed=EPISODE_SEED, task_id=TASK_ID)
    candidates = build_action_candidates(len(observation.coordinates))
    history: list[str] = []

    print(f"Task: {TASK_ID}")
    print(f"Episode: {env.state.episode_id}")
    print(f"Initial energy: {observation.energy:.3f}")
    print(f"Initial score: {float(observation.metadata.get('score', 0.0)):.3f}")

    for step in range(1, MAX_STEPS + 1):
        if observation.done:
            print("Environment signalled done. Stopping early.")
            break

        shortlisted = shortlist_candidates(env, candidates, SHORTLIST_SIZE)
        candidate_actions = [item[0] for item in shortlisted]
        user_prompt = build_user_prompt(observation, shortlisted, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"Model request failed ({exc}). Falling back to strongest heuristic action.")
            response_text = ""

        chosen_action = parse_action_response(response_text, candidate_actions)
        observation = env.step(chosen_action)

        reward = float(observation.reward or 0.0)
        history_line = (
            f"step={step} action={format_action(chosen_action)} "
            f"reward={reward:+.3f} energy={observation.energy:.3f} "
            f"score={float(observation.metadata.get('score', 0.0)):.3f}"
        )
        history.append(history_line)

        print(f"Step {step}: {format_action(chosen_action)}")
        print(
            f"  reward={reward:+.3f} "
            f"energy={observation.energy:.3f} "
            f"score={float(observation.metadata.get('score', 0.0)):.3f} "
            f"contacts={observation.hydrophobic_contacts} "
            f"collisions={observation.collisions}"
        )

        if observation.done:
            print("Episode complete.")
            break
    else:
        print(f"Reached max steps ({MAX_STEPS}).")

    print("Final summary:")
    print(f"  steps={observation.step_count}")
    print(f"  energy={observation.energy:.3f}")
    print(f"  score={float(observation.metadata.get('score', 0.0)):.3f}")
    print(f"  hydrophobic_contacts={observation.hydrophobic_contacts}")
    print(f"  collisions={observation.collisions}")


if __name__ == "__main__":
    main()
