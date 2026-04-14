# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Convert log files to ShareGPT format for multi-modal training.

This module extracts message history from logs and converts them to ShareGPT format,
which is suitable for training multi-modal LLMs. It handles both text and image content.
"""

import json
import os
import shutil
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime


def decode_base64_image(base64_string: str, output_path: Path) -> bool:
    """
    Decode a base64 image string and save it as a file.

    Args:
        base64_string: Base64-encoded image string (with or without data:image/...;base64, prefix)
        output_path: Path where the image file should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_data = base64_string.split(",", 1)[1]
        else:
            base64_data = base64_string

        # Decode base64
        image_data = base64.b64decode(base64_data)

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(image_data)

        return True
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return False


def get_mime_type_from_base64(base64_string: str) -> str:
    """
    Extract MIME type from base64 data URL.

    Args:
        base64_string: Base64 string with potential data URL prefix

    Returns:
        MIME type string (e.g., "image/jpeg") or default "image/jpeg"
    """
    if base64_string.startswith("data:image/"):
        # Extract MIME type from data URL
        mime_part = base64_string.split(";")[0]
        return mime_part.replace("data:", "")
    return "image/jpeg"


def get_file_extension_from_mime(mime_type: str) -> str:
    """
    Convert MIME type to file extension.

    Args:
        mime_type: MIME type string (e.g., "image/jpeg")

    Returns:
        File extension (e.g., ".jpg")
    """
    mime_to_ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    return mime_to_ext.get(mime_type, ".jpg")


def process_content_with_images(
    content: Any,
    images_dir: Path,
    task_id: str,
    msg_idx: int,
    image_counter: List[int],
    existing_images: List[str] = None
) -> Tuple[str, List[str]]:
    """
    Process message content, extract and save images, update content with <image> markers.

    Args:
        content: Message content (string, list, or dict)
        images_dir: Directory to save extracted images
        task_id: Task identifier for naming images
        msg_idx: Message index for naming images
        image_counter: Mutable list to track image count [current_count]
        existing_images: List of existing image file paths (if already saved)

    Returns:
        Tuple of (processed_content_with_markers, list_of_image_paths)
    """
    image_paths = []

    if isinstance(content, list):
        # Multi-modal content (list of content items)
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    # Check if we have existing images to use
                    if existing_images and image_counter[0] < len(existing_images):
                        # Use existing image
                        image_paths.append(existing_images[image_counter[0]])
                        image_counter[0] += 1
                    else:
                        # Try to extract from base64
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url and image_url != "<image>" and image_url.startswith("data:image"):
                            # Save image to file
                            mime_type = get_mime_type_from_base64(image_url)
                            ext = get_file_extension_from_mime(mime_type)
                            image_filename = f"{task_id}_msg{msg_idx}_img{image_counter[0]}{ext}"
                            image_path = images_dir / image_filename

                            if decode_base64_image(image_url, image_path):
                                image_paths.append(str(image_path))
                                image_counter[0] += 1
                            else:
                                print(f"Warning: Failed to decode and save image {image_filename}")

                    # Always add <image> marker
                    text_parts.append("<image>")
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))

        processed_content = " ".join(text_parts) if text_parts else ""

    elif isinstance(content, str):
        # Text content - check if it contains base64 image
        if content.startswith("data:image") and ";base64," in content:
            # Check if we have existing images to use
            if existing_images and image_counter[0] < len(existing_images):
                # Use existing image
                image_paths.append(existing_images[image_counter[0]])
                image_counter[0] += 1
                processed_content = "<image>"
            else:
                # Save image to file
                mime_type = get_mime_type_from_base64(content)
                ext = get_file_extension_from_mime(mime_type)
                image_filename = f"{task_id}_msg{msg_idx}_img{image_counter[0]}{ext}"
                image_path = images_dir / image_filename

                if decode_base64_image(content, image_path):
                    image_paths.append(str(image_path))
                    image_counter[0] += 1
                else:
                    print(f"Warning: Failed to decode and save image {image_filename}")

                processed_content = "<image>"
        else:
            processed_content = content
    else:
        # Other formats - convert to string
        processed_content = str(content)

    return processed_content, image_paths


def rebuild_mcp_system_prompt(system_prompt: str, tool_definitions: List[Dict[str, Any]]) -> str:
    """
    Rebuild system prompt with MCP tool definitions when they are missing.

    When use_tool_calls=True (OpenAI native function calling), the system prompt
    saved in task logs doesn't contain tool definitions because they were passed
    via the API `tools` parameter. This function injects the tool definitions
    back into the system prompt in MCP XML format so the training data is complete.

    Args:
        system_prompt: The original system prompt (possibly without tool definitions)
        tool_definitions: List of MCP server definitions with their tools

    Returns:
        System prompt with tool definitions injected
    """
    if not tool_definitions:
        return system_prompt

    # Check if tool definitions are already present
    if "## Server name:" in system_prompt:
        return system_prompt

    # Build tool definitions section
    tools_section = ""
    for server in tool_definitions:
        server_name = server.get("name", "")
        if not server_name:
            continue
        tools_section += f"\n## Server name: {server_name}\n"

        tools = server.get("tools", [])
        for tool in tools:
            if "error" in tool and "name" not in tool:
                continue
            tools_section += f"### Tool name: {tool.get('name', '')}\n"
            tools_section += f"Description: {tool.get('description', '')}\n"
            tools_section += f"Input JSON schema: {tool.get('schema', {})}\n"

    if not tools_section:
        return system_prompt

    # Build the MCP tool-use formatting instructions
    mcp_instructions = """# Tool-Use Formatting Instructions 

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description: 
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
"param1": "value1",
"param2": "value2 \\"escaped string\\""
}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:

"""

    # Inject MCP instructions + tool definitions before "# General Objective"
    anchor = "# General Objective"
    if anchor in system_prompt:
        insert_pos = system_prompt.index(anchor)
        system_prompt = (
            system_prompt[:insert_pos]
            + mcp_instructions
            + tools_section
            + "\n"
            + system_prompt[insert_pos:]
        )
    else:
        # Fallback: append at the end
        system_prompt += "\n" + mcp_instructions + tools_section

    return system_prompt


def convert_tool_calls_to_mcp_format(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Convert OpenAI-style tool_calls to MCP format string.

    Args:
        tool_calls: List of tool call dictionaries in OpenAI format

    Returns:
        MCP-formatted tool call string
    """
    mcp_tool_call_templates = []

    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        arguments = function.get("arguments", "{}")

        # Parse tool name to extract server and tool name
        # Format: server_name-tool_name
        if "-" in tool_name:
            parts = tool_name.rsplit("-", maxsplit=1)
            if len(parts) == 2:
                server_name, tool_name_only = parts
            else:
                server_name = tool_name
                tool_name_only = tool_name
        else:
            server_name = tool_name
            tool_name_only = tool_name

        # Try to parse arguments as JSON for pretty formatting
        try:
            arguments_json = json.loads(arguments)
            arguments_str = json.dumps(arguments_json, ensure_ascii=False)
        except:
            arguments_str = arguments

        mcp_tool_call_template = f"\n\n<use_mcp_tool>\n<server_name>{server_name}</server_name>\n<tool_name>{tool_name_only}</tool_name>\n<arguments>\n{arguments_str}\n</arguments>\n</use_mcp_tool>"

        mcp_tool_call_templates.append(mcp_tool_call_template)

    return "\n\n".join(mcp_tool_call_templates)


def convert_messages_to_sharegpt(
    messages: List[Dict[str, Any]], images_dir: Path, task_id: str, existing_images: List[str] = None
) -> Dict[str, Any]:
    """
    Convert message list to ShareGPT format.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        images_dir: Directory to save extracted images
        task_id: Task identifier for naming images
        existing_images: List of existing image file paths (if already saved)

    Returns:
        ShareGPT-formatted dictionary with 'conversations' and 'images' keys
        Format: {"conversations": [{"from": "human/gpt", "value": "..."}], "images": [...]}
    """
    # Role mapping for ShareGPT format
    role_mapping = {
        "user": "human",
        "assistant": "gpt",
        "tool": "human"  # Tool return results are represented as human messages
    }

    sharegpt_conversations = []
    all_image_paths = []
    image_counter = [0]  # Use list to allow mutation in nested function

    for msg_idx, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")

        # Map role to ShareGPT format
        # Keep system messages as "system" role
        sharegpt_role = role_mapping.get(role, role) if role != "system" else "system"

        # Handle tool_calls in assistant messages
        if role == "assistant" and "tool_calls" in message and message["tool_calls"]:
            # Get reasoning content if exists
            reasoning_content = message.get("reasoning_content", "")

            # Process content and extract images
            processed_content, image_paths = process_content_with_images(
                content, images_dir, task_id, msg_idx, image_counter, existing_images
            )

            # Convert tool_calls to MCP format
            tool_calls_str = convert_tool_calls_to_mcp_format(message["tool_calls"])

            # Build final content: <thought>\nreasoning\n</thought>content + tool_calls
            final_content_parts = []

            # Add reasoning content if present
            if reasoning_content:
                final_content_parts.append(f"<thought>\n{reasoning_content}\n</thought>")

            # Add main content
            if processed_content:
                final_content_parts.append(processed_content)

            # Add tool calls
            final_content_parts.append(tool_calls_str)

            # Join all parts
            final_content = "".join(final_content_parts)

            # Add to conversations if content is not empty
            if final_content.strip():
                sharegpt_conversations.append({
                    "from": sharegpt_role,
                    "value": final_content
                })

            # Collect all image paths
            all_image_paths.extend(image_paths)

        elif role == "tool":
            # Tool return results - include as human messages
            # Process content to extract images and replace with <image> markers
            processed_content, image_paths = process_content_with_images(
                content, images_dir, task_id, msg_idx, image_counter, existing_images
            )
            if processed_content:
                sharegpt_conversations.append({
                    "from": sharegpt_role,
                    "value": processed_content
                })
            all_image_paths.extend(image_paths)

        else:
            # Regular user or assistant messages
            # Process content and extract images
            processed_content, image_paths = process_content_with_images(
                content, images_dir, task_id, msg_idx, image_counter, existing_images
            )

            # Handle reasoning_content in assistant messages without tool_calls
            final_content = processed_content
            if role == "assistant" and "reasoning_content" in message and message["reasoning_content"]:
                reasoning_content = message["reasoning_content"]
                if final_content:
                    final_content = f"<thought>\n{reasoning_content}\n</thought>{final_content}"
                else:
                    final_content = f"<thought>\n{reasoning_content}\n</thought>"

            # Add to conversations if content is not empty
            if final_content:
                sharegpt_conversations.append({
                    "from": sharegpt_role,
                    "value": final_content
                })

            # Collect all image paths
            all_image_paths.extend(image_paths)

    return {
        "conversations": sharegpt_conversations,
        "images": all_image_paths
    }


def extract_and_save_sharegpt(
    log_data: Dict[str, Any],
    output_dir: Path,
    input_filename: str,
    original_log_path: str = None,
    fallback_tool_definitions: List[Dict[str, Any]] = None,
):
    """
    Extract message history from log data and save as ShareGPT format.

    Args:
        log_data: Log data dictionary
        output_dir: Output directory for ShareGPT JSON files
        input_filename: Input filename (without extension)
        original_log_path: Path to original log file (for loading existing images)
        fallback_tool_definitions: External tool definitions to use when the log file
            doesn't contain them (for old data collected with use_tool_calls=True)
    """
    # Try to find and use existing images directory
    existing_images = []
    images_dir = None  # Only create if needed for new base64 images

    if original_log_path:
        original_path = Path(original_log_path)
        original_dir = original_path.parent

        # Try multiple locations for images directory (in order of priority)
        possible_locations = [
            # 1. Same directory as the JSON file (for successful_logs case)
            original_dir / f"{input_filename}_images",
            # 2. New format: save_images/xxx_images (in same dir)
            original_dir / "save_images" / f"{input_filename}_images",
            # 3. Old format: xxx_images (in parent dir)
            original_dir.parent / f"{input_filename}_images",
            # 4. save_images in parent dir
            original_dir.parent / "save_images" / f"{input_filename}_images",
        ]

        for images_dir in possible_locations:
            if images_dir.exists() and images_dir.is_dir():
                # Load images from this directory
                for img_file in sorted(images_dir.iterdir()):
                    if img_file.is_file():
                        existing_images.append(str(img_file))
                if existing_images:
                    print(f"✓ Found {len(existing_images)} existing image(s) in: {images_dir}")
                    break

    # If no existing images found, use output_dir as fallback for any new base64 images
    if not images_dir:
        images_dir = output_dir

    # 1. Extract main_agent_message_history
    main_agent_history = log_data.get("main_agent_message_history", {})
    if main_agent_history and "message_history" in main_agent_history:
        main_messages = main_agent_history["message_history"]

        # Prepend system_prompt if it exists
        system_prompt = main_agent_history.get("system_prompt", "")
        tool_definitions = main_agent_history.get("tool_definitions", [])

        # Use fallback tool_definitions for old data that doesn't have them saved
        if not tool_definitions and fallback_tool_definitions:
            tool_definitions = fallback_tool_definitions

        # If tool_definitions are available, rebuild the full MCP system prompt
        # so training data includes tool definitions
        if system_prompt and tool_definitions:
            system_prompt = rebuild_mcp_system_prompt(system_prompt, tool_definitions)

        if system_prompt and main_messages:
            # Create a new messages list with system prompt first
            main_messages_with_system = [
                {"role": "system", "content": system_prompt}
            ] + main_messages
            main_messages = main_messages_with_system

        if main_messages:
            sharegpt_data = convert_messages_to_sharegpt(
                main_messages, images_dir, input_filename, existing_images
            )

            # Save main agent ShareGPT records
            main_output_file = output_dir / f"{input_filename}_main_agent_sharegpt.json"
            with open(main_output_file, "w", encoding="utf-8") as f:
                json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

            print(f"✓ Saved main agent ShareGPT: {main_output_file}")
            print(f"  - Conversations: {len(sharegpt_data['conversations'])}")
            print(f"  - Images: {len(sharegpt_data['images'])}")

    # 2. Extract sub_agent_message_history_sessions
    sub_agent_sessions = log_data.get("sub_agent_message_history_sessions", {})
    if sub_agent_sessions:
        for session_name, session_data in sub_agent_sessions.items():
            if "message_history" in session_data:
                sub_messages = session_data["message_history"]
                if sub_messages:
                    # Rebuild sub-agent system prompt with tool definitions if needed
                    sub_system_prompt = session_data.get("system_prompt", "")
                    sub_tool_definitions = session_data.get("tool_definitions", [])
                    if not sub_tool_definitions and fallback_tool_definitions:
                        sub_tool_definitions = fallback_tool_definitions
                    if sub_system_prompt and sub_tool_definitions:
                        sub_system_prompt = rebuild_mcp_system_prompt(sub_system_prompt, sub_tool_definitions)
                    if sub_system_prompt:
                        sub_messages = [
                            {"role": "system", "content": sub_system_prompt}
                        ] + sub_messages
                    # Create separate images directory for each sub-agent
                    sub_images_dir = images_dir / f"{input_filename}_{session_name}_images"

                    sharegpt_data = convert_messages_to_sharegpt(
                        sub_messages, sub_images_dir, f"{input_filename}_{session_name}", None
                    )

                    # Save sub agent ShareGPT records
                    sub_output_file = (
                        output_dir / f"{input_filename}_{session_name}_sharegpt.json"
                    )
                    with open(sub_output_file, "w", encoding="utf-8") as f:
                        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

                    print(f"✓ Saved sub agent {session_name} ShareGPT: {sub_output_file}")
                    print(f"  - Conversations: {len(sharegpt_data['conversations'])}")
                    print(f"  - Images: {len(sharegpt_data['images'])}")


def main():
    """Main function"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert log files to ShareGPT format for multi-modal training."
    )
    parser.add_argument("log_file_path", help="Path to the log JSON file")
    parser.add_argument("output_dir", nargs="?", default="sharegpt_output",
                        help="Output directory for ShareGPT files (default: sharegpt_output)")
    parser.add_argument("--tool-defs", default=None,
                        help="Path to a JSON file containing tool_definitions "
                             "(for old data missing tool definitions in the log)")
    args = parser.parse_args()

    log_file_path = Path(args.log_file_path)
    output_dir = Path(args.output_dir)

    # Check if input file exists
    if not log_file_path.exists():
        print(f"Error: Log file does not exist: {log_file_path}")
        sys.exit(1)

    # Load external tool definitions if provided
    fallback_tool_definitions = None
    if args.tool_defs:
        tool_defs_path = Path(args.tool_defs)
        if not tool_defs_path.exists():
            print(f"Error: Tool definitions file does not exist: {tool_defs_path}")
            sys.exit(1)
        with open(tool_defs_path, "r", encoding="utf-8") as f:
            fallback_tool_definitions = json.load(f)
        print(f"Loaded external tool definitions from: {tool_defs_path}")

    try:
        # Read log file
        print(f"Reading log file: {log_file_path}")
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        # Extract input filename (without extension)
        input_filename = log_file_path.stem

        # Extract and save ShareGPT format
        print(f"Converting to ShareGPT format to: {output_dir}")
        extract_and_save_sharegpt(
            log_data, output_dir, input_filename, str(log_file_path),
            fallback_tool_definitions=fallback_tool_definitions,
        )

        print("\n✓ ShareGPT conversion completed!")
        print(f"Output directory: {output_dir.absolute()}")

    except json.JSONDecodeError as e:
        print(f"Error: Cannot parse JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
