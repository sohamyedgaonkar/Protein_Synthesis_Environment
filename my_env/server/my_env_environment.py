# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code Reviewer Environment Implementation.

An environment for training agents to review and improve Python code by:
- Cloning GitHub repositories
- Running code and capturing errors/logs
- Proposing code changes
- Providing rewards for improvements
"""

import os
import shutil
import subprocess
import tempfile
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from uuid import uuid4
import re

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CodeReviewerAction, CodeReviewerObservation, CodeMetrics, ErrorInfo, LineEditAction, LineEdit
except ImportError:
    from models import CodeReviewerAction, CodeReviewerObservation, CodeMetrics, ErrorInfo, LineEditAction, LineEdit


class CodeReviewerEnvironment(Environment):
    """
    Code Reviewer Environment for training agents to improve Python code.
    
    This environment:
    - Clones Python repositories from GitHub
    - Executes main.py and captures errors/logs
    - Tracks code quality metrics
    - Rewards agents for fixing errors, removing unused imports, etc.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the code reviewer environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._repo_dir = None
        self._original_errors: List[Dict] = []
        self._previous_errors_count = 0
        self._previous_warnings_count = 0
        self._all_code_content = ""
        self._code_summary_file = None

    def reset(self, repo_url: str ="https://github.com/sohamyedgaonkar/Test_For_CodeFixing.git") -> CodeReviewerObservation:
        """
        Reset the environment with a new GitHub repository.
        
        Args:
            repo_url: GitHub repository URL to clone
            
        Returns:
            CodeReviewerObservation with initial state, code, and errors
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Clean up old repository if it exists
        if self._repo_dir and os.path.exists(self._repo_dir):
            shutil.rmtree(self._repo_dir)
        
        # Create new temporary directory for repository
        base_repo_dir = Path("Repo")
        base_repo_dir.mkdir(exist_ok=True)
        
        # Clean old repo folders (keep only the latest one)
        self._cleanup_old_repos(base_repo_dir)
        
        # Create directory for this episode
        self._repo_dir = base_repo_dir / str(self._state.episode_id)
        self._repo_dir.mkdir(exist_ok=True)
        
        # Clone repository
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(self._repo_dir)],
                capture_output=True,
                check=True,
                timeout=60
            )
        except subprocess.CalledProcessError as e:
            error_info = ErrorInfo(
                error_type="GitCloneError",
                error_message=f"Failed to clone repository: {e.stderr.decode()}",
                traceback=str(e)
            )
            return CodeReviewerObservation(
                repo_url=repo_url,
                all_code="",
                execution_logs="",
                errors=[error_info],
                code_metrics=CodeMetrics(),
                step_count=self._state.step_count,
                episode_id=self._state.episode_id,
                done=True,
                reward=-1.0,
            )
        except Exception as e:
            error_info = ErrorInfo(
                error_type="CloneException",
                error_message=str(e)
            )
            return CodeReviewerObservation(
                repo_url=repo_url,
                all_code="",
                execution_logs="",
                errors=[error_info],
                code_metrics=CodeMetrics(),
                step_count=self._state.step_count,
                episode_id=self._state.episode_id,
                done=True,
                reward=-1.0,
            )
        
        # Collect all code from repository
        self._all_code_content = self._collect_all_code(self._repo_dir)
        
        # Create code summary file
        self._code_summary_file = self._repo_dir / "code_summary.txt"
        with open(str(self._code_summary_file), "w") as f:
            f.write(self._all_code_content)
        
        # Analyze code metrics
        code_metrics = self._analyze_code(self._repo_dir)
        
        # Execute main.py and capture output
        execution_logs, errors = self._execute_main_py(self._repo_dir)
        self._original_errors = errors
        self._previous_errors_count = len(errors)
        self._previous_warnings_count = len(code_metrics.warnings)
        
        return CodeReviewerObservation(
            repo_url=repo_url,
            all_code=self._all_code_content,
            code_file_path=str(self._code_summary_file),
            execution_logs=execution_logs,
            errors=[ErrorInfo(**err) if isinstance(err, dict) else err for err in errors],
            code_metrics=code_metrics,
            step_count=self._state.step_count,
            episode_id=self._state.episode_id,
            done=False,
            reward=0.0,
            metadata={
                "repo_url": repo_url,
                "initial_error_count": len(errors),
                "initial_warning_count": len(code_metrics.warnings)
            }
        )

    def step(self, action) -> CodeReviewerObservation:  # type: ignore
        """
        Execute a step by applying code changes and evaluating improvements.
        
        Supports two types of actions:
        1. CodeReviewerAction: Provide entire modified file content
        2. LineEditAction: Provide targeted line-level edits (replace/add/delete)
        
        Args:
            action: CodeReviewerAction or LineEditAction containing modifications
            
        Returns:
            CodeReviewerObservation with results and reward
        """
        if self._repo_dir is None:
            error_info = ErrorInfo(
                error_type="StateError",
                error_message="Environment not initialized. Call reset() first."
            )
            return CodeReviewerObservation(
                execution_logs="",
                errors=[error_info],
                code_metrics=CodeMetrics(),
                done=True,
                reward=-1.0,
            )
        
        self._state.step_count += 1
        
        # Determine action type and apply changes accordingly
        try:
            if isinstance(action, LineEditAction):
                # Handle line-by-line edits
                success, error_msg = self._apply_line_edits(self._repo_dir, action.edits)
                if not success:
                    error_info = ErrorInfo(
                        error_type="EditError",
                        error_message=error_msg
                    )
                    return CodeReviewerObservation(
                        execution_logs="",
                        errors=[error_info],
                        code_metrics=CodeMetrics(),
                        step_count=self._state.step_count,
                        episode_id=self._state.episode_id,
                        done=False,
                        reward=-0.5,
                    )
                description = action.description
            else:
                # Handle full file replacement (CodeReviewerAction)
                file_path = self._repo_dir / action.file_path
                # Ensure directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(str(file_path), "w") as f:
                    f.write(action.modified_code)
                description = action.description
        except Exception as e:
            error_info = ErrorInfo(
                error_type="FileWriteError",
                error_message=f"Failed to apply changes: {str(e)}"
            )
            return CodeReviewerObservation(
                execution_logs="",
                errors=[error_info],
                code_metrics=CodeMetrics(),
                step_count=self._state.step_count,
                episode_id=self._state.episode_id,
                done=False,
                reward=-0.5,
            )
        
        # Update all code content
        self._all_code_content = self._collect_all_code(self._repo_dir)
        
        # Update code summary file
        if self._code_summary_file:
            with open(str(self._code_summary_file), "w") as f:
                f.write(self._all_code_content)
        
        # Analyze code metrics
        code_metrics = self._analyze_code(self._repo_dir)
        
        # Execute main.py and capture output
        execution_logs, errors = self._execute_main_py(self._repo_dir)
        
        # Calculate reward based on improvements
        reward = self._calculate_reward(
            errors,
            code_metrics,
            description
        )
        
        # Update state for next step
        self._previous_errors_count = len(errors)
        self._previous_warnings_count = len(code_metrics.warnings)
        
        # Prepare metadata based on action type
        if isinstance(action, LineEditAction):
            file_modified = f"{len(action.edits)} line edits"
            action_type = "LineEditAction"
        else:
            file_modified = action.file_path
            action_type = "CodeReviewerAction"
        
        return CodeReviewerObservation(
            repo_url="",
            all_code=self._all_code_content,
            code_file_path=str(self._code_summary_file) if self._code_summary_file else "",
            execution_logs=execution_logs,
            errors=[ErrorInfo(**err) if isinstance(err, dict) else err for err in errors],
            code_metrics=code_metrics,
            step_count=self._state.step_count,
            episode_id=self._state.episode_id,
            done=len(errors) == 0 and len(code_metrics.warnings) == 0,
            reward=reward,
            metadata={
                "change_description": description,
                "file_modified": file_modified,
                "action_type": action_type,
                "error_count": len(errors),
                "warning_count": len(code_metrics.warnings),
                "errors_fixed": max(0, len(self._original_errors) - len(errors))
            }
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.
        
        Returns:
            State with episode_id and step_count
        """
        return self._state

    def _cleanup_old_repos(self, base_repo_dir: Path) -> None:
        """Remove old repository folders, keeping only recent ones."""
        if not base_repo_dir.exists():
            return
        
        repo_folders = sorted(
            [f for f in base_repo_dir.iterdir() if f.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep only the last 3 repo folders
        for old_repo in repo_folders[3:]:
            try:
                shutil.rmtree(old_repo)
            except Exception:
                pass

    def _collect_all_code(self, repo_dir: Path) -> str:
        """Collect all Python code from the repository."""
        all_code = []
        
        for py_file in repo_dir.rglob("*.py"):
            # Skip __pycache__ and .git directories
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            
            try:
                relative_path = py_file.relative_to(repo_dir)
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                all_code.append(f"\n{'='*60}\nFile: {relative_path}\n{'='*60}\n{content}")
            except Exception as e:
                all_code.append(f"\nError reading {py_file}: {str(e)}\n")
        
        return "\n".join(all_code)

    def _analyze_code(self, repo_dir: Path) -> CodeMetrics:
        """Analyze code for unused imports and other metrics."""
        metrics = CodeMetrics()
        py_files = []
        
        for py_file in repo_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
            py_files.append(py_file)
        
        metrics.total_files = len(py_files)
        total_lines = 0
        
        for py_file in py_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                total_lines += len(content.split("\n"))
                
                # Parse AST to find unused imports
                try:
                    tree = ast.parse(content)
                    unused_imports = self._find_unused_imports(tree, content)
                    metrics.unused_imports.extend(unused_imports)
                except SyntaxError:
                    metrics.syntax_errors += 1
            except Exception:
                pass
        
        metrics.total_lines = total_lines
        return metrics

    def _find_unused_imports(self, tree: ast.AST, content: str) -> List[str]:
        """Find unused imports in Python code."""
        unused = []
        imported_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name != "*":
                        imported_names.add(name)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        unused = list(imported_names - used_names)
        return unused[:10]  # Return top 10 unused imports

    def _execute_main_py(self, repo_dir: Path) -> Tuple[str, List[Dict]]:
        """Execute main.py and capture output/errors."""
        main_py = repo_dir / "main.py"
        errors = []
        
        # If main.py not in root, search in subdirectories (up to 2 levels deep)
        if not main_py.exists():
            found = False
            # Search one level deep
            for item in repo_dir.iterdir():
                if item.is_dir() and (item / "main.py").exists():
                    main_py = item / "main.py"
                    found = True
                    break
            
            # Search two levels deep if still not found
            if not found:
                for item in repo_dir.iterdir():
                    if item.is_dir():
                        for subitem in item.iterdir():
                            if subitem.is_dir() and (subitem / "main.py").exists():
                                main_py = subitem / "main.py"
                                found = True
                                break
                    if found:
                        break
        
        if not main_py.exists():
            return "main.py not found in repository or any subdirectories (up to 2 levels deep)", []
        
        try:
            # Use the directory containing main.py as cwd
            main_cwd = main_py.parent
            
            # Use just the filename "main.py" with cwd set to the correct directory
            result = subprocess.run(
                ["python", "main.py"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(main_cwd)
            )
            
            execution_logs = result.stdout
            
            if result.returncode != 0:
                # Parse error from stderr
                stderr = result.stderr
                errors = self._parse_errors(stderr)
                execution_logs += f"\n\nSTDERR:\n{stderr}"
            
            return execution_logs, errors
            
        except subprocess.TimeoutExpired:
            errors.append({
                "error_type": "TimeoutError",
                "error_message": "main.py execution timed out after 30 seconds",
                "traceback": ""
            })
            return "Execution timeout", errors
        except Exception as e:
            errors.append({
                "error_type": "ExecutionError",
                "error_message": f"Failed to execute main.py: {str(e)}",
                "traceback": ""
            })
            return f"Error: {str(e)}", errors


    def _parse_errors(self, stderr: str) -> List[Dict]:
        """Parse error messages from stderr."""
        errors = []
        
        # Try to extract traceback information
        lines = stderr.split("\n")
        for i, line in enumerate(lines):
            if "Error" in line or "Exception" in line:
                errors.append({
                    "error_type": line.split(":")[0] if ":" in line else "Error",
                    "error_message": line,
                    "line_number": None,
                    "traceback": stderr[:500]  # First 500 chars of traceback
                })
        
        if not errors and stderr.strip():
            errors.append({
                "error_type": "RuntimeError",
                "error_message": stderr[:200],
                "line_number": None,
                "traceback": stderr
            })
        
        return errors

    def _calculate_reward(
        self,
        current_errors: List[Dict],
        code_metrics: CodeMetrics,
        description: str
    ) -> float:
        """Calculate reward based on improvements made."""
        reward = 0.0
        
        # Track error shifts (error moved from one line to another = fixed + new error)
        # Count unique error messages to detect when errors are actually fixed
        current_error_types = set(e.get('error_type', '') for e in current_errors if isinstance(e, dict))
        current_error_types.update(e.error_type for e in current_errors if hasattr(e, 'error_type'))
        
        original_error_types = set(e.get('error_type', '') for e in self._original_errors if isinstance(e, dict))
        original_error_types.update(e.error_type for e in self._original_errors if hasattr(e, 'error_type'))
        
        # Errors that were present before but are now gone (fixed)
        errors_type_fixed = len(original_error_types - current_error_types)
        
        # Directly count error count reduction (handles error shifting)
        error_count_reduction = max(0, len(self._original_errors) - len(current_errors))
        
        # Use the direct count reduction for rewarding
        # This rewards when errors shift (line 50 -> line 100) as it's still one less error
        reward += error_count_reduction * 0.5
        
        # Additional bonus for fixing specific error types (stacked reward)
        reward += min(errors_type_fixed * 0.2, 0.5)  # Cap at 0.5
        
        # Reward for removing unused imports
        unused_count = len(code_metrics.unused_imports)
        imports_removed = max(0, len(self._original_errors) - unused_count)
        reward += max(imports_removed * 0.1, 0)
        
        # Reward for reducing warnings
        if len(code_metrics.warnings) < self._previous_warnings_count:
            reward += (self._previous_warnings_count - len(code_metrics.warnings)) * 0.1
        
        # Bonus for successful execution (no errors)
        if len(current_errors) == 0 and len(code_metrics.syntax_errors) == 0:
            reward += 0.3  # Increased bonus
        
        # Penalty for syntax errors
        if code_metrics.syntax_errors > 0:
            reward -= code_metrics.syntax_errors * 0.2
        
        # Penalty for introducing new errors
        if len(current_errors) > self._previous_errors_count:
            reward -= 0.1
        
        return reward

    def _apply_line_edits(self, repo_dir: Path, edits: List) -> Tuple[bool, str]:
        """
        Apply line-level edits to files.
        
        Args:
            repo_dir: Repository directory
            edits: List of LineEdit objects
            
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Group edits by file
            edits_by_file: Dict[str, List] = {}
            for edit in edits:
                if edit.file_path not in edits_by_file:
                    edits_by_file[edit.file_path] = []
                edits_by_file[edit.file_path].append(edit)
            
            # Apply edits to each file
            for file_path, file_edits in edits_by_file.items():
                target_file = repo_dir / file_path
                
                # Ensure directory exists
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read file if it exists
                if target_file.exists():
                    with open(target_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                else:
                    lines = []
                
                # Sort edits by line number in descending order to avoid index shifts
                # Delete operations first (highest line numbers), then add/replace
                sorted_edits = sorted(
                    file_edits,
                    key=lambda e: (e.line_number, 0 if e.operation == "delete" else 1),
                    reverse=True
                )
                
                # Helper function to detect indentation
                def get_indentation(line: str) -> str:
                    """Extract leading whitespace from a line."""
                    return line[:len(line) - len(line.lstrip())]
                
                # Helper function to preserve indentation when adding lines
                def apply_indentation(new_code: str, source_line_num: int) -> str:
                    """Apply proper indentation to new code based on context."""
                    # Get indentation from surrounding lines
                    indent = ""
                    
                    # Try to get indentation from previous line
                    if source_line_num > 0 and source_line_num - 1 < len(lines):
                        indent = get_indentation(lines[source_line_num - 1])
                    # Try to get indentation from next line
                    elif source_line_num < len(lines):
                        indent = get_indentation(lines[source_line_num])
                    
                    # Split new_code into lines and apply indentation
                    code_lines = new_code.split('\n')
                    indented_lines = []
                    
                    for i, line in enumerate(code_lines):
                        if i == 0:
                            # First line: use as-is or apply detected indent if it looks like code
                            if line.strip() and not line[0].isspace():
                                indented_lines.append(indent + line)
                            else:
                                indented_lines.append(line)
                        else:
                            # Subsequent lines: preserve relative indentation
                            if line.strip():
                                indented_lines.append(indent + line.lstrip())
                            else:
                                indented_lines.append(line)
                    
                    return '\n'.join(indented_lines)
                
                # Apply each edit
                for edit in sorted_edits:
                    line_num = edit.line_number - 1  # Convert to 0-indexed
                    
                    if edit.operation == "replace":
                        # Replace a line (line must exist)
                        if 0 <= line_num < len(lines):
                            # Apply indentation to replacement
                            new_code = apply_indentation(edit.new_code, line_num)
                            if not new_code.endswith("\n"):
                                new_code += "\n"
                            lines[line_num] = new_code
                        else:
                            return False, f"Line {edit.line_number} does not exist in {file_path}"
                    
                    elif edit.operation == "add":
                        # Add a line at specified position (inserts before the line number)
                        new_code = apply_indentation(edit.new_code, line_num)
                        if not new_code.endswith("\n"):
                            new_code += "\n"
                        # Insert at line_num position
                        lines.insert(line_num, new_code)
                    
                    elif edit.operation == "delete":
                        # Delete a line (line must exist)
                        if 0 <= line_num < len(lines):
                            lines.pop(line_num)
                        else:
                            return False, f"Line {edit.line_number} does not exist in {file_path}"
                    
                    else:
                        return False, f"Unknown operation: {edit.operation}. Must be 'replace', 'add', or 'delete'"
                
                # Write the modified content back
                with open(target_file, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            
            return True, "All edits applied successfully"
            
        except Exception as e:
            return False, f"Error applying line edits: {str(e)}"

