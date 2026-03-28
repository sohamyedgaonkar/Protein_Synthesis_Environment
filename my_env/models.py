# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Code Reviewer Environment.

Models for code review actions and observations with detailed error tracking,
code analysis, and performance metrics.
"""

from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CodeReviewerAction(Action):
    """Action for the Code Reviewer environment - proposed code changes."""

    file_path: str = Field(..., description="Path to the file being modified")
    modified_code: str = Field(..., description="The modified code content")
    description: str = Field(default="", description="Description of changes made")


class LineEdit(Action):
    """Represents a single line edit operation."""

    file_path: str = Field(..., description="Path to the file being modified")
    line_number: int = Field(..., description="Line number (1-indexed) to modify")
    operation: str = Field(..., description="Operation type: 'replace', 'add', or 'delete'")
    new_code: Optional[str] = Field(default="", description="New code content (for 'replace' and 'add' operations)")


class LineEditAction(Action):
    """Action for granular line-by-line code modifications."""

    edits: List[LineEdit] = Field(..., description="List of line edit operations")
    description: str = Field(default="", description="Description of changes made")


class ErrorInfo(Observation):
    """Information about an error encountered during code execution."""

    error_type: str = Field(..., description="Type of error (SyntaxError, ImportError, etc)")
    error_message: str = Field(..., description="Error message")
    line_number: Optional[int] = Field(default=None, description="Line number where error occurred")
    traceback: str = Field(default="", description="Full traceback")


class CodeMetrics(Observation):
    """Metrics about the code quality and performance."""

    unused_imports: List[str] = Field(default_factory=list, description="List of unused imports")
    syntax_errors: int = Field(default=0, description="Number of syntax errors")
    runtime_errors: int = Field(default=0, description="Number of runtime errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    total_lines: int = Field(default=0, description="Total lines of code")
    total_files: int = Field(default=0, description="Total number of Python files")


class CodeReviewerObservation(Observation):
    """Observation from the Code Reviewer environment."""

    repo_url: str = Field(default="", description="Repository URL being reviewed")
    all_code: str = Field(default="", description="All code from the repository")
    code_file_path: str = Field(default="", description="Path to the code summary file")
    execution_logs: str = Field(default="", description="Logs from executing main.py")
    errors: List[ErrorInfo] = Field(default_factory=list, description="List of errors encountered")
    code_metrics: CodeMetrics = Field(default_factory=CodeMetrics, description="Code metrics and quality info")
    step_count: int = Field(default=0, description="Current step count")
    episode_id: str = Field(default="", description="Episode ID")
    done: bool = Field(default=False, description="Whether the episode is done")
    reward: float = Field(default=0.0, description="Reward for this step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
