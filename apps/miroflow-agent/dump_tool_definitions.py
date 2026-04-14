# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Dump tool definitions from MCP servers to a JSON file.

This script connects to MCP servers based on the agent config (e.g., lmm-searcher.yaml),
fetches tool definitions (names, descriptions, schemas), and saves them to a JSON file.
The output can be used by process_logs.py to rebuild system prompts for old trace data
that was collected with use_tool_calls=True but lacks saved tool_definitions.

Usage:
    uv run python dump_tool_definitions.py agent=lmm-searcher
    uv run python dump_tool_definitions.py agent=lmm-searcher hydra.run.dir=/tmp/dump
"""

import asyncio
import json
import sys

import hydra
from miroflow_tools.manager import ToolManager
from omegaconf import DictConfig

from src.config.settings import create_mcp_server_parameters, expose_sub_agents_as_tools


async def fetch_and_dump(cfg: DictConfig):
    """Fetch tool definitions from MCP servers and dump to JSON."""
    main_agent_mcp_configs, main_agent_blacklist = create_mcp_server_parameters(
        cfg, cfg.agent.main_agent
    )
    tool_manager = ToolManager(main_agent_mcp_configs, tool_blacklist=main_agent_blacklist)

    print(f"Fetching tool definitions from {len(main_agent_mcp_configs)} MCP server(s)...")
    tool_definitions = await tool_manager.get_all_tool_definitions()

    if cfg.agent.sub_agents is not None:
        tool_definitions += expose_sub_agents_as_tools(cfg.agent.sub_agents)

    output_path = "tool_definitions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tool_definitions, f, ensure_ascii=False, indent=2)

    total_tools = sum(len(s.get("tools", [])) for s in tool_definitions)
    print(f"Saved {len(tool_definitions)} server(s), {total_tools} tool(s) to {output_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(fetch_and_dump(cfg))


if __name__ == "__main__":
    main()
