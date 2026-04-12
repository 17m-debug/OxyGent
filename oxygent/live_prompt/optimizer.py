"""Prompt optimization module for improving prompt quality.

This module provides intelligent prompt optimization capabilities using LLM-based
analysis and improvement strategies. It supports framework-specific constraints
and custom optimization requirements.
"""

import logging
import json
from typing import Optional, Dict, Any, List
from ..schemas import OxyRequest


logger = logging.getLogger(__name__)


# Framework-specific constraint templates
FRAMEWORK_CONSTRAINTS = {
    "react": {
        "required_format": """
The prompt MUST include these exact instructions for tool calling format:

3. When you need to use a tool, you must only respond with the exact JSON object format below, nothing else:
```json
{
    "think": "Your thinking (if analysis is needed)",
    "tool_name": "Tool name",
    "arguments": {
        "parameter_name": "parameter_value"
    }
}
```

And this instruction for sequential tool calling:
If solving the user's problem requires multiple tool calls, call only one tool at a time.
""",
        "validation_rules": [
            "Must contain the exact JSON format specification",
            "Must include 'call only one tool at a time' instruction",
            "Must preserve the ${tools_description} placeholder if present",
            "Must preserve the ${additional_prompt} placeholder if present"
        ]
    },
    "general": {
        "required_format": "",
        "validation_rules": [
            "Clear and concise instructions",
            "Proper structure and formatting",
            "Appropriate for the use case"
        ]
    }
}


# Base optimization prompt template
OPTIMIZATION_PROMPT_TEMPLATE = """
You are an expert prompt engineer specializing in improving prompt quality and effectiveness.

## Current Prompt to Optimize:
```
{current_prompt}
```

## Optimization Goals:
{optimization_goals}

## Framework-Specific Constraints:
{framework_constraints}

## User's Custom Requirements:
{custom_requirements}

## Task:
Analyze the current prompt and provide an optimized version that:

1. **Clarity**: Makes instructions more clear and unambiguous
2. **Structure**: Improves organization and flow
3. **Completeness**: Ensures all necessary components are present
4. **Effectiveness**: Enhances the prompt's ability to generate desired outputs
5. **Compliance**: Follows all framework-specific constraints

## Response Format:
Provide your response in the following JSON format:
```json
{{
    "analysis": "Your analysis of the current prompt's strengths and weaknesses",
    "improvements": ["List of specific improvements made", "each as a separate item"],
    "optimized_prompt": "The complete optimized prompt text",
    "rationale": "Explanation of why these improvements will lead to better results",
    "validation": {{
        "meets_constraints": true/false,
        "missing_elements": ["Any required elements that are still missing"],
        "warnings": ["Any potential issues or concerns"]
    }}
}}
```

Ensure the optimized prompt is production-ready and follows best practices in prompt engineering.
"""


class PromptOptimizer:
    """Intelligent prompt optimization using LLM-based analysis.

    This class provides functionality to analyze and improve prompts based on
    framework-specific requirements, user goals, and prompt engineering best practices.

    Attributes:
        llm_class: The LLM class name to use for optimization (auto-detected)
        optimization_strategies: Dictionary of predefined optimization strategies
    """

    # Predefined optimization strategies
    OPTIMIZATION_STRATEGIES = {
        "clarity": """
- Improve clarity and specificity of instructions
- Remove ambiguous language
- Add concrete examples where helpful
- Simplify complex phrasing
""",
        "structure": """
- Improve logical flow and organization
- Group related instructions together
- Use clear section headings
- Enhance readability with proper formatting
""",
        "completeness": """
- Add missing essential components
- Ensure all necessary context is provided
- Include edge case handling
- Specify desired output format
""",
        "effectiveness": """
- Strengthen task-oriented instructions
- Add role/context setting
- Improve constraint specification
- Enhance output quality expectations
""",
        "comprehensive": """
Combine all optimization strategies: clarity, structure, completeness, and effectiveness.
This provides a full prompt engineering review and improvement.
"""
    }

    def __init__(self, llm_model: Optional[str] = None):
        """Initialize the PromptOptimizer.

        Args:
            llm_model: Optional LLM model to use. If not provided, will auto-detect
                     from registered LLM classes.
        """
        self.llm_model = self._auto_detect_llm(llm_model)

    def _auto_detect_llm(self, preferred_llm: Optional[str] = None) -> str:
        """Auto-detect available LLM instance name.

        Args:
            preferred_llm: Preferred LLM instance name if provided

        Returns:
            The LLM instance name to use

        Raises:
            ValueError: If no LLM instance is found
        """
        # If specific LLM is provided, validate it exists
        if preferred_llm:
            available_instances = self._get_available_llm_instances()
            if preferred_llm in available_instances:
                logger.info(f"Using specified LLM: {preferred_llm}")
                return preferred_llm
            else:
                logger.warning(f"Specified LLM '{preferred_llm}' not found in registered instances")
                logger.info("Attempting to auto-detect available LLM...")

        # Try to find registered LLM instances
        available_instances = self._get_available_llm_instances()

        if available_instances:
            # Use the first available LLM instance
            selected_llm = available_instances[0]
            logger.info(f"Auto-detected LLM instance: {selected_llm}")
            return selected_llm

        # If no instances found, raise error
        raise ValueError(
            f"No LLM instances found. "
            f"Please register an LLM instance (e.g., oxy.HttpLLM(name='default_llm', ...)) "
            f"or specify llm_model parameter with an existing instance name."
        )

    def _get_available_llm_instances(self) -> List[str]:
        """Get list of registered LLM instance names from global MAS.

        Returns:
            List of LLM instance names that are registered
        """
        try:
            # Import routes to access global MAS instance
            from .. import routes

            # Get global MAS instance
            if not hasattr(routes, '_global_mas_instance') or routes._global_mas_instance is None:
                logger.debug("Global MAS instance not available")
                return []

            mas = routes._global_mas_instance

            # Filter for LLM instances from oxy_name_to_oxy
            llm_instances = [
                name for name, instance in mas.oxy_name_to_oxy.items()
                if 'LLM' in instance.__class__.__name__
            ]

            logger.debug(f"Found LLM instances: {llm_instances}")
            return llm_instances

        except Exception as e:
            logger.debug(f"Error getting LLM instances: {e}")
            return []

    async def optimize(
        self,
        current_prompt: str,
        agent_type: str = "general",
        optimization_strategy: str = "comprehensive",
        custom_requirements: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize a prompt based on specified strategy and constraints.

        Args:
            current_prompt: The original prompt to optimize
            agent_type: Type of agent (e.g., "react", "general")
            optimization_strategy: Strategy to use for optimization
            custom_requirements: Additional user requirements
            context: Additional context information (e.g., use_case, target_audience, constraints)

        Returns:
            Dictionary containing optimization results with keys:
                - analysis: Analysis of the original prompt
                - improvements: List of improvements made
                - optimized_prompt: The optimized prompt text
                - rationale: Explanation of improvements
                - validation: Validation results
        """
        try:
            # Get optimization goals
            optimization_goals = self._get_optimization_goals(optimization_strategy)

            # Get framework constraints
            framework_constraints = self._get_framework_constraints(agent_type)

            # Enhance custom requirements with context if provided
            enhanced_requirements = custom_requirements
            if context:
                context_info = []
                if context.get("use_case"):
                    context_info.append(f"Use case: {context['use_case']}")
                if context.get("target_audience"):
                    context_info.append(f"Target audience: {context['target_audience']}")
                if context.get("constraints"):
                    context_info.append(f"Additional constraints: {context['constraints']}")

                if context_info:
                    enhanced_requirements = f"{custom_requirements}\n\nContext:\n" + "\n".join(context_info)

            # Build optimization prompt
            optimizer_prompt = self._build_optimization_prompt(
                current_prompt=current_prompt,
                optimization_goals=optimization_goals,
                framework_constraints=framework_constraints,
                custom_requirements=enhanced_requirements
            )

            # Execute optimization via LLM
            result = await self._execute_optimization(optimizer_prompt)

            # Validate result only if optimization succeeded
            if result and result.get("optimized_prompt"):
                try:
                    result["validated"] = self._validate_optimized_prompt(
                        result["optimized_prompt"],
                        agent_type
                    )
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")
                    result["validated"] = {
                        "meets_constraints": False,
                        "missing_elements": ["Validation error"],
                        "warnings": [f"Validation failed: {str(e)}"]
                    }
            elif result:
                # Optimized prompt is None or missing
                result["validated"] = {
                    "meets_constraints": False,
                    "missing_elements": ["No optimized prompt generated"],
                    "warnings": ["Optimization did not produce a result"]
                }

            return result

        except Exception as e:
            logger.error(f"Error during prompt optimization: {e}")
            return {
                "error": str(e),
                "optimized_prompt": current_prompt,  # Fallback to original
                "analysis": "Optimization failed, returning original prompt",
                "improvements": [],
                "rationale": f"Optimization error: {str(e)}",
                "validation": {
                    "meets_constraints": False,
                    "missing_elements": ["Optimization failed"],
                    "warnings": [str(e)]
                }
            }

    def _get_optimization_goals(self, strategy: str) -> str:
        """Get optimization goals based on strategy.

        Args:
            strategy: The optimization strategy name

        Returns:
            String describing optimization goals
        """
        return self.OPTIMIZATION_STRATEGIES.get(
            strategy,
            self.OPTIMIZATION_STRATEGIES["comprehensive"]
        )

    def _get_framework_constraints(self, agent_type: str) -> str:
        """Get framework-specific constraints.

        Args:
            agent_type: Type of agent

        Returns:
            String describing required format and constraints
        """
        constraints = FRAMEWORK_CONSTRAINTS.get(agent_type, FRAMEWORK_CONSTRAINTS["general"])

        output = []
        if constraints.get("required_format"):
            output.append(constraints["required_format"])

        if constraints.get("validation_rules"):
            output.append("\nValidation Rules:")
            for rule in constraints["validation_rules"]:
                output.append(f"- {rule}")

        return "\n".join(output) if output else "No specific constraints for this agent type."

    def _build_optimization_prompt(
        self,
        current_prompt: str,
        optimization_goals: str,
        framework_constraints: str,
        custom_requirements: str
    ) -> str:
        """Build the optimization prompt for the LLM.

        Args:
            current_prompt: Original prompt text
            optimization_goals: Optimization goals description
            framework_constraints: Framework constraints
            custom_requirements: User custom requirements

        Returns:
            Complete optimization prompt
        """
        return OPTIMIZATION_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            optimization_goals=optimization_goals,
            framework_constraints=framework_constraints,
            custom_requirements=custom_requirements or "None specified"
        )

    async def _execute_optimization(self, optimizer_prompt: str) -> Dict[str, Any]:
        """Execute the optimization using LLM.

        Args:
            optimizer_prompt: The optimization prompt to send to LLM

        Returns:
            Parsed optimization result as dictionary
        """
        try:
            # Import routes to access global MAS instance
            from .. import routes

            # Get global MAS instance
            if not hasattr(routes, '_global_mas_instance') or routes._global_mas_instance is None:
                raise ValueError("MAS instance not available. Please ensure the service is running.")

            mas = routes._global_mas_instance

            # Get the registered LLM instance by name
            if self.llm_model not in mas.oxy_name_to_oxy:
                raise ValueError(f"LLM instance '{self.llm_model}' not found in MAS")

            llm = mas.oxy_name_to_oxy[self.llm_model]

            # Create request with messages format for OpenAILLM
            # Wrap the optimizer prompt as a system message
            messages = [
                {"role": "system", "content": "You are an expert prompt engineering assistant."},
                {"role": "user", "content": optimizer_prompt}
            ]

            # Create request with messages parameter
            oxy_request = OxyRequest(
                query=optimizer_prompt,  # For compatibility
                arguments={"messages": messages}  # OpenAILLM requires this
            )

            # Execute
            oxy_response = await llm.execute(oxy_request)

            # Parse JSON response
            result = self._parse_optimization_result(oxy_response.output)

            return result

        except Exception as e:
            logger.error(f"Error executing LLM optimization: {e}")
            raise

    def _parse_optimization_result(self, output: str) -> Dict[str, Any]:
        """Parse the LLM output into structured result.

        Args:
            output: Raw LLM output

        Returns:
            Parsed dictionary with optimization results
        """
        try:
            # Handle empty or None output
            if not output or not isinstance(output, str):
                return {
                    "analysis": "LLM returned empty or invalid output",
                    "improvements": [],
                    "optimized_prompt": None,
                    "rationale": f"Invalid output type: {type(output)}",
                    "validation": {
                        "meets_constraints": False,
                        "missing_elements": ["Failed to generate optimized prompt"],
                        "warnings": ["LLM output was empty or None"]
                    }
                }

            # Try to extract JSON from the output
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # No JSON found, return the raw output as the optimized prompt
                    return {
                        "analysis": "LLM did not return structured JSON, using raw output",
                        "improvements": ["Generated optimized prompt"],
                        "optimized_prompt": output,
                        "rationale": "The LLM returned a text response instead of JSON format. Using the raw output as the optimized prompt.",
                        "validation": {
                            "meets_constraints": False,
                            "missing_elements": ["JSON validation skipped"],
                            "warnings": ["Response was not in JSON format"]
                        }
                    }

            result = json.loads(json_str)

            # Ensure required fields are present
            required_fields = ["analysis", "improvements", "optimized_prompt", "rationale"]
            for field in required_fields:
                if field not in result:
                    result[field] = f"[Missing {field}]"

            if "validation" not in result:
                result["validation"] = {
                    "meets_constraints": True,
                    "missing_elements": [],
                    "warnings": []
                }

            return result

        except json.JSONDecodeError as e:
            # JSON parsing failed, return raw output
            logger.warning(f"Failed to parse JSON from LLM output: {e}")
            return {
                "analysis": "Failed to parse JSON response",
                "improvements": [],
                "optimized_prompt": output if output else None,
                "rationale": f"JSON parsing error: {str(e)}",
                "validation": {
                    "meets_constraints": False,
                    "missing_elements": ["JSON parsing failed"],
                    "warnings": ["Using raw output instead"]
                }
            }
        except Exception as e:
            logger.error(f"Error parsing optimization result: {e}")
            # Return structured error response
            return {
                "analysis": "Failed to parse LLM response",
                "improvements": [],
                "optimized_prompt": output if output else None,
                "rationale": f"Parsing error: {str(e)}",
                "validation": {
                    "meets_constraints": False,
                    "missing_elements": ["Failed to generate optimized prompt"],
                    "warnings": ["LLM output could not be parsed"]
                }
            }

    def _validate_optimized_prompt(self, optimized_prompt: str, agent_type: str) -> Dict[str, Any]:
        """Validate that the optimized prompt meets framework requirements.

        Args:
            optimized_prompt: The optimized prompt to validate
            agent_type: Type of agent

        Returns:
            Validation result dictionary
        """
        # Handle None or empty prompt
        if not optimized_prompt:
            return {
                "meets_constraints": False,
                "missing_elements": ["No prompt to validate"],
                "warnings": ["Optimized prompt is empty or None"]
            }

        constraints = FRAMEWORK_CONSTRAINTS.get(agent_type, {})
        validation_rules = constraints.get("validation_rules", [])

        validation_result = {
            "meets_constraints": True,
            "missing_elements": [],
            "warnings": []
        }

        for rule in validation_rules:
            # Check if the rule is satisfied
            if "Must contain" in rule or "Must include" in rule:
                # Extract what needs to be contained
                keyword = rule.lower()
                if "json format" in keyword.lower():
                    if "```json" not in optimized_prompt and '{"think"' not in optimized_prompt:
                        validation_result["missing_elements"].append(rule)
                        validation_result["meets_constraints"] = False

                elif "one tool at a time" in keyword.lower():
                    if "one tool at a time" not in optimized_prompt.lower():
                        validation_result["missing_elements"].append(rule)
                        validation_result["meets_constraints"] = False

                elif "${tools_description}" in rule:
                    if "${tools_description}" not in optimized_prompt:
                        validation_result["missing_elements"].append(rule)
                        validation_result["meets_constraints"] = False

                elif "${additional_prompt}" in rule:
                    if "${additional_prompt}" not in optimized_prompt:
                        validation_result["warnings"].append(rule)

        return validation_result

    def get_available_strategies(self) -> List[str]:
        """Get list of available optimization strategies.

        Returns:
            List of strategy names
        """
        return list(self.OPTIMIZATION_STRATEGIES.keys())

    def get_supported_agent_types(self) -> List[str]:
        """Get list of supported agent types.

        Returns:
            List of agent type names
        """
        return list(FRAMEWORK_CONSTRAINTS.keys())


# Singleton instance
_optimizer_instance: Optional[PromptOptimizer] = None


def get_prompt_optimizer(llm_model: Optional[str] = None, force_new: bool = False) -> PromptOptimizer:
    """Get or create the singleton PromptOptimizer instance.

    Args:
        llm_model: Optional LLM model to use. If not provided, will auto-detect.
                   If singleton exists and llm_model differs, a new instance is created.
        force_new: Force creation of a new instance even if singleton exists

    Returns:
        PromptOptimizer instance

    Note:
        The LLM is auto-detected on first call. Subsequent calls reuse the same
        instance unless force_new=True or a different llm_model is specified.
    """
    global _optimizer_instance

    if force_new or _optimizer_instance is None:
        _optimizer_instance = PromptOptimizer(llm_model=llm_model)
        logger.info(f"Created new PromptOptimizer with LLM: {_optimizer_instance.llm_model}")
    elif llm_model is not None and llm_model != _optimizer_instance.llm_model:
        # Different LLM requested, create new instance
        _optimizer_instance = PromptOptimizer(llm_model=llm_model)
        logger.info(f"Recreated PromptOptimizer with LLM: {_optimizer_instance.llm_model}")

    return _optimizer_instance
