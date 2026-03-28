#!/usr/bin/env python3
"""
Simple Local AI Inference Agent - No HTTP, Direct Environment Usage

This agent runs locally without any HTTP server:
- Directly uses CodeReviewerEnvironment
- Analyzes code errors with OpenAI
- Applies fixes locally
- Iterates until errors are fixed
"""

import os
import json
import sys
import re
from typing import Dict, Any, List, Optional

# Import environment and models
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "my_env")))
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "my_env", "server")))
try:
    from my_env.server.my_env_environment import CodeReviewerEnvironment
    from my_env.models import CodeReviewerAction, LineEditAction, LineEdit
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the Code_Reviewer directory")
    exit(1)

try:
    import openai
except ImportError:
    print("Error: openai package not installed")
    print("Install with: pip install openai")
    exit(1)


class SimpleLocalAIAgent:
    """Simple AI agent that runs locally without HTTP."""

    def __init__(self, openai_model: str = "gpt-4-turbo", max_iterations: int = 5, verbose: bool = True):
        """
        Initialize the local AI agent.
        
        Args:
            openai_model: OpenAI model to use
            max_iterations: Maximum iterations
            verbose: Print logs
        """
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Set it with: $env:OPENAI_API_KEY = 'sk-...' (PowerShell)")
            exit(1)
        
        self.model = openai_model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.environment = CodeReviewerEnvironment()
        self.client = openai.OpenAI(api_key=api_key)

    def log(self, message: str):
        """Print log if verbose."""
        if self.verbose:
            print(f"[INFO] {message}")

    def get_suggestions_from_openai(self, errors: List[Dict], code: str, logs: str) -> str:
        """Ask OpenAI for fix suggestions."""
        self.log(f"Calling OpenAI {self.model} for analysis...")
        
        # Convert ErrorInfo objects to dictionaries
        errors_dicts = []
        for error in errors[:5]:  # Limit to 5 errors
            if hasattr(error, 'model_dump'):
                errors_dicts.append(error.model_dump())
            elif hasattr(error, 'dict'):
                errors_dicts.append(error.dict())
            else:
                errors_dicts.append(error)
        
        prompt = f"""You are an expert Python code reviewer and debugger.

## Current Errors (showing up to 5):
{json.dumps(errors_dicts, indent=2)}

## Execution Logs:
{logs[:500] if logs else "No logs"}

## Code (first 1500 chars):
{code[:1500]}

## Task:
Analyze these errors and suggest specific fixes. Format your response ONLY as valid JSON:
{{
    "analysis": "Brief explanation of issues",
    "fixes": [
        {{
            "line_number": <int>,
            "operation": "replace|add|delete",
            "new_code": "code to add/replace (empty for delete)",
            "reason": "Why this fixes the issue"
        }}
    ]
}}

IMPORTANT: Only return valid JSON, no other text."""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    def parse_fixes(self, response: str) -> List[Dict]:
        """Extract fixes from AI response."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            return data.get("fixes", [])
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("fixes", [])
            except json.JSONDecodeError:
                pass
        
        self.log(f"Could not parse fixes from response: {response[:200]}")
        return []

    def create_action_from_fixes(self, fixes: List[Dict]) -> Optional[LineEditAction]:
        """Convert fixes to LineEditAction."""
        if not fixes:
            return None
        
        edits = []
        for fix in fixes:
            edit = LineEdit(
                file_path="main.py",
                line_number=fix.get("line_number", 0),
                operation=fix.get("operation", "replace"),
                new_code=fix.get("new_code", "")
            )
            edits.append(edit)
        
        return LineEditAction(edits=edits)

    def run_episode(self, repo_url: str) -> Dict[str, Any]:
        """Run one complete analysis episode."""
        self.log(f"\n{'='*60}")
        self.log(f"Starting analysis of: {repo_url}")
        self.log(f"{'='*60}\n")
        
        # Reset environment
        self.log("Resetting environment and cloning repository...")
        result = self.environment.reset(repo_url=repo_url)
        observation = result
        
        initial_errors = len(observation.errors)
        self.log(f"Initial state:")
        self.log(f"  - Errors: {initial_errors}")
        self.log(f"  - Files: {observation.code_metrics.total_files}")
        self.log(f"  - Lines: {observation.code_metrics.total_lines}")
        
        total_reward = 0.0
        errors_fixed = 0
        
        # Iterate
        for iteration in range(self.max_iterations):
            self.log(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            current_errors = len(observation.errors)
            
            if current_errors == 0:
                self.log("✓ All errors fixed!")
                break
            
            self.log(f"Current errors: {current_errors}")
            
            # Get suggestions from OpenAI
            response = self.get_suggestions_from_openai(
                observation.errors,
                observation.all_code,
                observation.execution_logs
            )
            
            # Parse fixes
            fixes = self.parse_fixes(response)
            self.log(f"AI suggested {len(fixes)} fixes")
            
            if not fixes:
                self.log("No fixes suggested, stopping")
                break
            
            # Create action
            action = self.create_action_from_fixes(fixes)
            if not action:
                break
            
            # Apply action
            self.log(f"Applying {len(action.edits)} edits...")
            step_result = self.environment.step(action)
            observation = step_result
            
            reward = step_result.reward
            total_reward += reward
            
            new_errors = len(observation.errors)
            fixed_in_iteration = current_errors - new_errors
            errors_fixed += fixed_in_iteration
            
            # Analyze error types to show shifts
            prev_error_types = set(e.error_type if hasattr(e, 'error_type') else e.get('error_type', '(unknown)') for e in self.environment._original_errors)
            current_error_types = set(e.error_type if hasattr(e, 'error_type') else e.get('error_type', '(unknown)') for e in observation.errors)
            errors_fixed_types = len(prev_error_types - current_error_types)
            
            self.log(f"✓ Step completed:")
            self.log(f"  - Reward: {reward:.3f} (total: {total_reward:.3f})")
            self.log(f"  - Errors fixed: {fixed_in_iteration}")
            self.log(f"  - Error types fixed: {errors_fixed_types}")
            self.log(f"  - Errors remaining: {new_errors}")
            
            if observation.errors:
                error_summary = ", ".join(set(
                    (e.error_type if hasattr(e, 'error_type') else e.get('error_type', '?'))
                    for e in observation.errors[:3]
                ))
                self.log(f"  - Remaining error types: {error_summary}")
        
        final_errors = len(observation.errors)
        
        results = {
            "repo_url": repo_url,
            "initial_errors": initial_errors,
            "final_errors": final_errors,
            "errors_fixed": initial_errors - final_errors,
            "total_reward": total_reward,
            "iterations": min(self.max_iterations, iteration + 1) if 'iteration' in locals() else 0,
        }
        
        # Summary
        self.log(f"\n{'='*60}")
        self.log("EPISODE SUMMARY")
        self.log(f"{'='*60}")
        self.log(f"Initial Errors:     {results['initial_errors']}")
        self.log(f"Final Errors:       {results['final_errors']}")
        self.log(f"Errors Fixed:       {results['errors_fixed']}")
        self.log(f"Total Reward:       {results['total_reward']:.3f}")
        self.log(f"Iterations:         {results['iterations']}")
        
        if results['initial_errors'] > 0:
            fix_rate = (results['errors_fixed'] / results['initial_errors']) * 100
            self.log(f"Fix Success Rate:   {fix_rate:.1f}%")
        
        self.log(f"{'='*60}\n")
        
        return results


def main():
    """Main entry point."""
    # Test repository
    test_repo = "https://github.com/sohamyedgaonkar/Test_For_CodeFixing.git"
    
    # Create agent
    agent = SimpleLocalAIAgent(
        openai_model="gpt-4-turbo",
        max_iterations=5,
        verbose=True
    )
    
    # Run analysis
    results = agent.run_episode(test_repo)
    
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
