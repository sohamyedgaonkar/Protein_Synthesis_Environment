# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code Reviewer Environment Client."""

from typing import Dict, Optional, Union

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    CodeReviewerAction,
    CodeReviewerObservation,
    CodeMetrics,
    ErrorInfo,
    LineEditAction,
    LineEdit
)


class CodeReviewerEnv(
    EnvClient[CodeReviewerAction, CodeReviewerObservation, State]
):
    """
    Client for the Code Reviewer Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CodeReviewerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(repo_url="https://github.com/user/repo")
        ...     print(result.observation.execution_logs)
        ...
        ...     result = client.step(CodeReviewerAction(
        ...         file_path="main.py",
        ...         modified_code="improved code here",
        ...         description="Fixed bug in line 42"
        ...     ))
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodeReviewerEnv.from_docker_image("code-reviewer-env:latest")
        >>> try:
        ...     result = client.reset(repo_url="https://github.com/user/repo")
        ...     print(result.observation.errors)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: Union[CodeReviewerAction, LineEditAction]) -> Dict:
        """
        Convert action to JSON payload for step message.

        Args:
            action: CodeReviewerAction or LineEditAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        if isinstance(action, LineEditAction):
            return {
                "edits": [
                    {
                        "file_path": edit.file_path,
                        "line_number": edit.line_number,
                        "operation": edit.operation,
                        "new_code": edit.new_code,
                    }
                    for edit in action.edits
                ],
                "description": action.description,
            }
        else:
            # CodeReviewerAction
            return {
                "file_path": action.file_path,
                "modified_code": action.modified_code,
                "description": action.description,
            }

    def _parse_result(self, payload: Dict) -> StepResult[CodeReviewerObservation]:
        """
        Parse server response into StepResult[CodeReviewerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CodeReviewerObservation
        """
        obs_data = payload.get("observation", {})
        
        # Parse errors
        errors = []
        for err in obs_data.get("errors", []):
            errors.append(ErrorInfo(**err))
        
        # Parse code metrics
        metrics_data = obs_data.get("code_metrics", {})
        code_metrics = CodeMetrics(**metrics_data)
        
        observation = CodeReviewerObservation(
            repo_url=obs_data.get("repo_url", ""),
            all_code=obs_data.get("all_code", ""),
            code_file_path=obs_data.get("code_file_path", ""),
            execution_logs=obs_data.get("execution_logs", ""),
            errors=errors,
            code_metrics=code_metrics,
            step_count=obs_data.get("step_count", 0),
            episode_id=obs_data.get("episode_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
