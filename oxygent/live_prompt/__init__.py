"""
Live Prompt Management Module
Provides real-time prompt management and hot-reload functionality for OxyGent agents
"""

# Core ES-based prompt management
from .manager import get_prompt_manager, get_dynamic_prompt, PromptManager

# Hot-reload functionality
from .wrapper import (
    setup_dynamic_agents,
    hot_reload_prompt,
    hot_reload_agent,
    hot_reload_all_prompts,
    dynamic_agent_manager,
)

# Version synchronization for multi-instance cache consistency
from .version import (
    VersionSyncCoordinator,
    get_version_sync_coordinator,
    start_version_sync,
    stop_version_sync,
)

# Prompt optimization
from .optimizer import get_prompt_optimizer, PromptOptimizer

__all__ = [
    # Core ES-based prompt management
    'get_prompt_manager',
    'get_dynamic_prompt',
    'PromptManager',

    # Hot-reload functionality
    'setup_dynamic_agents',
    'hot_reload_prompt',
    'hot_reload_agent',
    'hot_reload_all_prompts',
    'dynamic_agent_manager',

    # Version synchronization
    'VersionSyncCoordinator',
    'get_version_sync_coordinator',
    'start_version_sync',
    'stop_version_sync',

    # Prompt optimization
    'get_prompt_optimizer',
    'PromptOptimizer',
]