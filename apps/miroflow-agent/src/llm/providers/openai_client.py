# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
OpenAI-compatible LLM client implementation.

This module provides the OpenAIClient class for interacting with OpenAI's API
and OpenAI-compatible endpoints (such as vLLM, Qwen, DeepSeek, etc.).

Features:
- Async and sync API support
- Automatic retry with exponential backoff
- Token usage tracking and context length management
- MCP tool call parsing and response processing
"""

import asyncio
import dataclasses
import logging
from typing import Any, Dict, List, Tuple, Union
import os

import tiktoken
from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AzureOpenAI,
    DefaultAsyncHttpxClient,
    DefaultHttpxClient,
    OpenAI,
)

from ...utils.prompt_utils import generate_mcp_system_prompt, generate_openai_function_calling_system_prompt
from ..base_client import BaseClient

logger = logging.getLogger("miroflow_agent")


@dataclasses.dataclass
class OpenAIClient(BaseClient):
    def _create_client(self) -> Union[AsyncOpenAI, OpenAI, AsyncAzureOpenAI, AzureOpenAI]:
        """Create LLM client"""
        http_client_args = {"headers": {"x-upstream-session-id": self.task_id}}

        openai_backend = (os.environ.get("OPENAI_BACKEND") or "").strip().lower()
        use_azure = openai_backend == "azure"

        if use_azure:
            azure_endpoint = self.cfg.llm.get("azure_endpoint") or self.base_url
            api_version = self.cfg.llm.get("api_version") or os.environ.get("OPENAI_API_VERSION")

            if not azure_endpoint:
                raise ValueError(
                    "OPENAI_BACKEND=azure requires `llm.azure_endpoint` (or `llm.base_url`)."
                )
            if not api_version:
                raise ValueError(
                    "OPENAI_BACKEND=azure requires `llm.api_version` (or env OPENAI_API_VERSION)."
                )

            if self.async_client:
                return AsyncAzureOpenAI(
                    api_key=self.api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    http_client=DefaultAsyncHttpxClient(**http_client_args),
                )
            return AzureOpenAI(
                api_key=self.api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                http_client=DefaultHttpxClient(**http_client_args),
            )

        if self.async_client:
            return AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=DefaultAsyncHttpxClient(**http_client_args),
            )
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=DefaultHttpxClient(**http_client_args),
        )

    def _update_token_usage(self, usage_data: Any) -> None:
        """Update cumulative token usage"""
        if usage_data:
            input_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage_data, "completion_tokens", 0) or 0
            prompt_tokens_details = getattr(usage_data, "prompt_tokens_details", None)
            if prompt_tokens_details:
                cached_tokens = (
                    getattr(prompt_tokens_details, "cached_tokens", None) or 0
                )
            else:
                cached_tokens = 0

            # Record token usage for the most recent call
            self.last_call_tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }

            # OpenAI does not provide cache_creation_input_tokens
            self.token_usage["total_input_tokens"] += input_tokens
            self.token_usage["total_output_tokens"] += output_tokens
            self.token_usage["total_cache_read_input_tokens"] += cached_tokens

            self.task_log.log_step(
                "info",
                "LLM | Token Usage",
                f"Input: {self.token_usage['total_input_tokens']}, "
                f"Output: {self.token_usage['total_output_tokens']}",
            )

    async def _create_message(
        self,
        system_prompt: str,
        messages_history: List[Dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
    ):
        """
        Send message to OpenAI API.
        :param system_prompt: System prompt string.
        :param messages_history: Message history list.
        :return: OpenAI API response object or None (if error occurs).
        """

        # Create a copy for sending to LLM (to avoid modifying the original)
        messages_for_llm = [m.copy() for m in messages_history]

        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            # Check if there's already a system or developer message
            if messages_for_llm and messages_for_llm[0]["role"] in [
                "system",
                "developer",
            ]:
                messages_for_llm[0] = {
                    "role": "system",
                    "content": system_prompt,
                }

            else:
                messages_for_llm.insert(
                    0,
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                )

        # Filter tool results to save tokens (only affects messages sent to LLM)
        messages_for_llm = self._remove_tool_result_from_messages(
            messages_for_llm, keep_tool_result
        )

        # Sanitize: some API proxies (e.g. Gemini gateway) reject messages
        # where content is None or "".  Assistant messages WITH tool_calls
        # having empty content is normal (the model is just calling tools
        # without saying anything), so we leave those alone — only patch
        # truly broken cases (empty content without tool_calls).
        for msg in messages_for_llm:
            content = msg.get("content")
            is_empty = content is None or (isinstance(content, str) and content.strip() == "")
            if not is_empty:
                continue
            role = msg.get("role", "")
            if role == "assistant" and not msg.get("tool_calls"):
                msg["content"] = "..."
            elif role == "tool":
                msg["content"] = "No output."
            elif role == "user":
                msg["content"] = "..."

        # Retry loop with dynamic max_tokens adjustment
        max_retries = 100
        base_wait_time = 2
        max_length_limit_retries = 3
        current_max_tokens = self.max_tokens

        # Convert MCP tool definitions to OpenAI function calling format if enabled
        openai_tools = None
        if self.use_tool_calls and tools_definitions:
            openai_tools = self.convert_tool_definition_to_openai_format(tools_definitions)
            self.task_log.log_step(
                "info",
                "LLM | Function Calling",
                f"Using OpenAI native function calling with {len(openai_tools)} tools",
            )

        for attempt in range(max_retries):
            params = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": messages_for_llm,
                "stream": False,
                "top_p": self.top_p,
                "extra_body": {},
            }
            
            # Add tools parameter for native function calling
            if openai_tools:
                params["tools"] = openai_tools
                params["tool_choice"] = "auto"
            
            # Check if the model is GPT-5, and adjust the parameter accordingly
            if "gpt-5" in self.model_name:
                # Use 'max_completion_tokens' for GPT-5
                params["max_completion_tokens"] = current_max_tokens
            else:
                # Use 'max_tokens' for GPT-4 and other models
                params["max_tokens"] = current_max_tokens

            # Add repetition_penalty if it's not the default value
            if self.repetition_penalty != 1.0:
                params["extra_body"]["repetition_penalty"] = self.repetition_penalty

            if "deepseek-v3-1" in self.model_name:
                params["extra_body"]["thinking"] = {"type": "enabled"}

            # auto-detect if we need to continue from the last assistant message
            if messages_for_llm and messages_for_llm[-1].get("role") == "assistant":
                params["extra_body"]["continue_final_message"] = True
                params["extra_body"]["add_generation_prompt"] = False

            try:
                if self.async_client:
                    response = await self.client.chat.completions.create(**params)
                else:
                    response = self.client.chat.completions.create(**params)
                # Update token count
                self._update_token_usage(getattr(response, "usage", None))

                # Fallback: if API returned None for token counts, estimate via tiktoken
                usage = getattr(response, "usage", None)
                if usage and getattr(usage, "prompt_tokens", None) is None:
                    estimated_input = self._estimate_messages_tokens(messages_for_llm)
                    resp_text = getattr(response.choices[0].message, "content", "") or ""
                    estimated_output = self._estimate_tokens(resp_text)
                    self.last_call_tokens = {
                        "prompt_tokens": estimated_input,
                        "completion_tokens": estimated_output,
                    }
                    logger.warning(
                        f"API returned None for token counts, using tiktoken fallback: "
                        f"input≈{estimated_input}, output≈{estimated_output}"
                    )

                self.task_log.log_step(
                    "info",
                    "LLM | Response Status",
                    f"{getattr(response.choices[0], 'finish_reason', 'N/A')}",
                )

                # Check if response was truncated due to length limit
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                if finish_reason == "length":
                    # If this is not the last retry, increase max_tokens and retry
                    if attempt < max_length_limit_retries - 1:
                        # Increase max_tokens by 10%
                        current_max_tokens = int(current_max_tokens * 1.2)
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached",
                            f"Response was truncated due to length limit (attempt {attempt + 1}/{max_retries}). Increasing max_tokens to {current_max_tokens} and retrying...",
                        )
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        # Last retry, return the truncated response instead of raising exception
                        self.task_log.log_step(
                            "warning",
                            "LLM | Length Limit Reached - Returning Truncated Response",
                            f"Response was truncated after {max_retries} attempts. Returning truncated response to allow ReAct loop to continue.",
                        )
                        # Return the truncated response and let the orchestrator handle it
                        return response, messages_history

                # Check if the last 50 characters of the response appear more than 5 times in the response content.
                # If so, treat it as a severe repeat and trigger a retry.
                if hasattr(response.choices[0], "message") and hasattr(
                    response.choices[0].message, "content"
                ):
                    resp_content = response.choices[0].message.content or ""
                else:
                    resp_content = getattr(response.choices[0], "text", "")

                if resp_content and len(resp_content) >= 50:
                    tail_50 = resp_content[-50:]
                    repeat_count = resp_content.count(tail_50)
                    if repeat_count > 5:
                        # If this is not the last retry, retry
                        if attempt < max_retries - 1:
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected",
                                f"Severe repeat: the last 50 chars appeared over 5 times (attempt {attempt + 1}/{max_retries}), retrying...",
                            )
                            await asyncio.sleep(base_wait_time)
                            continue
                        else:
                            # Last retry, return anyway
                            self.task_log.log_step(
                                "warning",
                                "LLM | Repeat Detected - Returning Anyway",
                                f"Severe repeat detected after {max_retries} attempts. Returning response anyway.",
                            )

                # Success - return the original messages_history (not the filtered copy)
                # This ensures that the complete conversation history is preserved in logs
                return response, messages_history

            except asyncio.TimeoutError as e:
                if attempt < max_retries - 1:
                    self.task_log.log_step(
                        "warning",
                        "LLM | Timeout Error",
                        f"Timeout error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...",
                    )
                    await asyncio.sleep(base_wait_time)
                    continue
                else:
                    self.task_log.log_step(
                        "error",
                        "LLM | Timeout Error",
                        f"Timeout error after {max_retries} attempts: {str(e)}",
                    )
                    raise e
            except asyncio.CancelledError as e:
                self.task_log.log_step(
                    "error",
                    "LLM | Request Cancelled",
                    f"Request was cancelled: {str(e)}",
                )
                raise e
            except Exception as e:
                if "Error code: 400" in str(e) and "longer than the model" in str(e):
                    self.task_log.log_step(
                        "error",
                        "LLM | Context Length Error",
                        f"Error: {str(e)}",
                    )
                    raise e
                else:
                    error_str = str(e)
                    # Non-retryable errors: raise immediately instead of wasting retries
                    if "high risk" in error_str:
                        self.task_log.log_step(
                            "error",
                            "LLM | API Error",
                            "request was rejected because it was considered high risk"
                        )
                        raise e

                    if attempt < max_retries - 1:
                        self.task_log.log_step(
                            "warning",
                            "LLM | API Error",
                            f"Error (attempt {attempt + 1}/{max_retries}): {error_str}, retrying...",
                        )
                        # Debug: Print messages content for content-related errors
                        should_dump_messages = (
                            "InvalidParameter" in error_str
                            or "Invalid base64" in error_str
                            or "-4003" in error_str
                            or "至少要包含" in error_str
                            or "cannot identify image" in error_str
                        )
                        if should_dump_messages:
                            self._debug_dump_messages(messages_for_llm, error_str)
                            # -4003 is a content validation error, retrying won't help
                            if "-4003" in error_str or "至少要包含" in error_str:
                                self.task_log.log_step(
                                    "error",
                                    "LLM | Content Validation Error (-4003)",
                                    "Messages contain empty content or unreachable image URLs. "
                                    "Check debug logs above for details. Not retrying.",
                                )
                                raise e
                            if "cannot identify image" in error_str:
                                self.task_log.log_step(
                                    "error",
                                    "LLM | Invalid Image Error",
                                    "Messages contain an image that cannot be identified (corrupted or invalid format). "
                                    "Check debug logs above for details. Not retrying.",
                                )
                                raise e
                        await asyncio.sleep(base_wait_time)
                        continue
                    else:
                        self.task_log.log_step(
                            "error",
                            "LLM | API Error",
                            f"Error after {max_retries} attempts: {str(e)}",
                        )
                        raise e

        # Should never reach here, but just in case
        raise Exception("Unexpected error: retry loop completed without returning")

    def _debug_dump_messages(self, messages_for_llm: List[Dict], error_str: str) -> None:
        """Dump full message contents for debugging API errors like -4003 / InvalidParameter."""
        logger.error(f"=== Debug: Messages dump for error: {error_str[:200]} ===")
        logger.error(f"Total messages: {len(messages_for_llm)}")
        for idx, msg in enumerate(messages_for_llm):
            role = msg.get("role", "unknown")
            content = msg.get("content")
            tool_call_id = msg.get("tool_call_id", None)

            # Flag empty / None content
            if content is None:
                logger.error(f"Message {idx} [{role}]: ⚠️ content is None!")
                if tool_call_id:
                    logger.error(f"  tool_call_id={tool_call_id}, name={msg.get('name', 'N/A')}")
                continue
            if isinstance(content, str) and content.strip() == "":
                logger.error(f"Message {idx} [{role}]: ⚠️ content is EMPTY string! (len={len(content)})")
                if tool_call_id:
                    logger.error(f"  tool_call_id={tool_call_id}, name={msg.get('name', 'N/A')}")
                continue
            if isinstance(content, list) and len(content) == 0:
                logger.error(f"Message {idx} [{role}]: ⚠️ content is EMPTY list!")
                continue

            if isinstance(content, list):
                logger.error(f"Message {idx} [{role}]: (multimodal, {len(content)} items)")
                for item_idx, item in enumerate(content):
                    item_type = item.get("type", "unknown")
                    if item_type == "image_url":
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = str(image_url)
                        logger.error(f"  Item {item_idx} [image_url]: {url[:200]}...")
                        if url.startswith("data:"):
                            try:
                                if "," in url:
                                    header, b64_data = url.split(",", 1)
                                    logger.error(f"    -> Header: {header}, Base64 length: {len(b64_data)}")
                                    if len(b64_data) < 100:
                                        logger.error(f"    -> WARNING: Base64 data too short!")
                                    import base64 as b64_module
                                    try:
                                        decoded = b64_module.b64decode(b64_data)
                                        logger.error(f"    -> Decoded size: {len(decoded)} bytes, First 20 bytes: {decoded[:20]}")
                                        if decoded[:20].strip().lower().startswith((b'<!doctype', b'<html', b'<head', b'<?xml')):
                                            logger.error(f"    -> ERROR: Decoded content is HTML, not an image!")
                                    except Exception as decode_err:
                                        logger.error(f"    -> ERROR: Failed to decode base64: {decode_err}")
                                else:
                                    logger.error(f"    -> WARNING: No comma found in data URL")
                            except Exception as parse_err:
                                logger.error(f"    -> ERROR parsing data URL: {parse_err}")
                        elif not url.startswith("http"):
                            logger.error(f"    -> WARNING: URL doesn't start with http or data: scheme")
                    elif item_type == "text":
                        text = item.get("text", "")
                        if not text or text.strip() == "":
                            logger.error(f"  Item {item_idx} [text]: ⚠️ EMPTY text item!")
                        else:
                            logger.error(f"  Item {item_idx} [text]: {text[:300]}...")
                    else:
                        logger.error(f"  Item {item_idx} [{item_type}]: {str(item)[:200]}...")
            else:
                content_preview = str(content)[:1000] if len(str(content)) > 1000 else str(content)
                logger.error(f"Message {idx} [{role}]: {content_preview}")

            # Log tool_calls if present on assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                tc_list = msg["tool_calls"]
                logger.error(f"  tool_calls ({len(tc_list)}): {[tc.get('id', tc.get('function', {}).get('name', 'N/A')) for tc in tc_list]}")

        logger.error("=== End Debug ===")

    def process_llm_response(
        self, llm_response: Any, message_history: List[Dict], agent_type: str = "main"
    ) -> tuple[str, bool, List[Dict]]:
        """Process LLM response"""
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            self.task_log.log_step(
                "error", "LLM | Response Error", f"Error: {error_msg}"
            )
            return "", True, message_history  # Exit loop, return message_history

        finish_reason = llm_response.choices[0].finish_reason
        message = llm_response.choices[0].message
        
        # Extract reasoning_content if present (for Kimi/DeepSeek thinking models)
        reasoning_content = getattr(message, "reasoning_content", None)
        if not reasoning_content:
            reasoning_content = getattr(message, "reasoning", None)
        
        # Extract LLM response text
        if finish_reason == "stop":
            assistant_response_text = message.content or ""

            assistant_message = {"role": "assistant", "content": assistant_response_text}
            # Preserve reasoning_content for multi-turn conversation context
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            message_history.append(assistant_message)

        elif finish_reason == "tool_calls":
            # OpenAI native function calling - model wants to call tools
            assistant_response_text = message.content or ""
            tool_calls = message.tool_calls
            
            # Explicitly construct the dict to guarantee tool_calls are
            # preserved.  model_dump(exclude_unset=True) can silently drop
            # tool_calls depending on SDK version / proxy behaviour, leaving
            # an assistant message with only empty content and triggering
            # -4003 on the next API call.
            # Ref: Seed1.8 cookbook directly appends the raw response dict.
            assistant_message = {
                "role": "assistant",
                "content": assistant_response_text,
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
            
            # Preserve reasoning_content for multi-turn conversation context
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            
            message_history.append(assistant_message)
            
            # Log tool_calls for debugging
            if tool_calls:
                for tc in tool_calls:
                    logger.info(f"Tool call stored: id={tc.id}, name={tc.function.name}")
            
            self.task_log.log_step(
                "info",
                "LLM | Tool Calls",
                f"Model requested {len(tool_calls) if tool_calls else 0} tool call(s)",
            )

        elif finish_reason == "length":
            assistant_response_text = message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty."
            elif "Context length exceeded" in assistant_response_text:
                # This is the case where context length is exceeded, needs special handling
                self.task_log.log_step(
                    "warning",
                    "LLM | Context Length",
                    "Detected context length exceeded, returning error status",
                )
                assistant_message = {"role": "assistant", "content": assistant_response_text}
                if reasoning_content:
                    assistant_message["reasoning_content"] = reasoning_content
                message_history.append(assistant_message)
                return (
                    assistant_response_text,
                    True,
                    message_history,
                )  # Return True to indicate need to exit loop

            # Add assistant response to history
            assistant_message = {"role": "assistant", "content": assistant_response_text}
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            message_history.append(assistant_message)

        else:
            raise ValueError(
                f"Unsupported finish reason: {finish_reason}"
            )

        return assistant_response_text, False, message_history

    def extract_tool_calls_info(
        self, llm_response: Any, assistant_response_text: str
    ) -> List[Dict]:
        """Extract tool call information from LLM response.
        
        Supports two modes:
        1. Native OpenAI function calling: tool_calls are in llm_response.choices[0].message.tool_calls
        2. MCP XML format: tool_calls are embedded in assistant_response_text as <use_mcp_tool> tags
        """
        from ...utils.parsing_utils import parse_llm_response_for_tool_calls
        
        # Check if native function calling was used (tool_calls in response)
        if (llm_response and 
            llm_response.choices and 
            hasattr(llm_response.choices[0].message, 'tool_calls') and
            llm_response.choices[0].message.tool_calls):
            # Native OpenAI function calling format
            tool_calls = llm_response.choices[0].message.tool_calls
            return parse_llm_response_for_tool_calls(tool_calls)
        
        # Fall back to MCP XML format parsing from text
        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(
        self, message_history: List[Dict], all_tool_results_content_with_id: List[Tuple]
    ) -> List[Dict]:
        """
        Update message history with tool calls data (llm client specific).

        Handles both text-only and multi-modal content formats.
        Supports two modes:
        1. OpenAI native function calling: uses 'tool' role with tool_call_id and name
        2. MCP XML format: uses 'user' role with merged content
        
        Args:
            message_history: List of message dictionaries
            all_tool_results_content_with_id: List of tuples (call_id, tool_name, tool_result_content)
        """
        # Check if we're using native function calling (check if any tool_call has an id)
        has_tool_call_ids = any(
            item[0] is not None for item in all_tool_results_content_with_id
        )
        
        # Debug logging
        logger.info(f"update_message_history: use_tool_calls={self.use_tool_calls}, "
                    f"has_tool_call_ids={has_tool_call_ids}, "
                    f"num_results={len(all_tool_results_content_with_id)}")
        for i, item in enumerate(all_tool_results_content_with_id):
            logger.info(f"  Result {i}: call_id={item[0]}, tool_name={item[1]}")
        
        if has_tool_call_ids and self.use_tool_calls:
            # OpenAI native function calling format
            # Each tool result needs its own message with 'tool' role
            # Images are included directly in the tool message content

            for item in all_tool_results_content_with_id:
                tool_call_id = item[0]
                tool_name = item[1]
                tool_result_content = item[2]

                if isinstance(tool_result_content, dict):
                    if tool_result_content.get("type") == "text":
                        content = tool_result_content.get("text", "")
                    else:
                        content = str(tool_result_content)
                elif isinstance(tool_result_content, list):
                    has_images = any(
                        ci.get("type") == "image_url" for ci in tool_result_content
                    )
                    if has_images:
                        content = tool_result_content
                    else:
                        content = "\n".join(
                            ci.get("text", "") for ci in tool_result_content
                            if ci.get("type") == "text"
                        )
                else:
                    content = str(tool_result_content)

                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content,
                })

                if isinstance(content, list):
                    img_count = sum(1 for ci in content if ci.get("type") == "image_url")
                    logger.info(f"Tool message for {tool_name} includes {img_count} image(s) directly")

            return message_history
        
        # MCP XML format: use 'user' role with merged content
        # Collect all content items
        content_items = []

        for item in all_tool_results_content_with_id:
            # item is (call_id, tool_name, tool_result_content)
            tool_result_content = item[2]

            # Handle both dict and list formats
            if isinstance(tool_result_content, dict):
                # Text-only format
                if tool_result_content.get("type") == "text":
                    content_items.append(tool_result_content)
            elif isinstance(tool_result_content, list):
                # Multi-modal format (list of content items)
                content_items.extend(tool_result_content)

        # Check if we have any image content
        has_images = any(
            item.get("type") == "image_url" for item in content_items
        )

        if has_images:
            # Multi-modal format: keep as list
            message_history.append(
                {
                    "role": "user",
                    "content": content_items,
                }
            )
        else:
            # Text-only format: merge into single string
            merged_text = "\n".join(
                [
                    item["text"]
                    for item in content_items
                    if item.get("type") == "text"
                ]
            )
            message_history.append(
                {
                    "role": "user",
                    "content": merged_text,
                }
            )

        return message_history

    def generate_agent_system_prompt(self, date: Any, mcp_servers: List[Dict]) -> str:
        """Generate system prompt based on tool calling mode.
        
        When use_tool_calls is True (native function calling), returns a simplified
        system prompt without tool definitions (tools are passed via API parameter).
        
        When use_tool_calls is False (MCP XML format), returns the full MCP system
        prompt with tool definitions embedded.
        """
        if self.use_tool_calls:
            # Native function calling: tools are passed via API, not in system prompt
            return generate_openai_function_calling_system_prompt(date)
        else:
            # MCP XML format: tools are embedded in system prompt
            return generate_mcp_system_prompt(date, mcp_servers)

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate the number of tokens in text"""
        if not hasattr(self, "encoding"):
            # Initialize tiktoken encoder
            try:
                self.encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                try:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.encoding = None

        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                self.task_log.log_step(
                    "error",
                    "LLM | Token Estimation Error",
                    f"Error: {str(e)}",
                )

        return len(text) // 4

    def _estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """Estimate total token count for a list of messages via tiktoken.
        Used as fallback when the API doesn't report usage (e.g. some proxy endpoints).
        """
        total = 0
        for m in messages:
            total += 4  # per-message overhead
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        total += self._estimate_tokens(str(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        total += self._estimate_image_tokens(url)
            else:
                total += self._estimate_tokens(str(content))
        return total

    def _estimate_image_tokens(self, image_url: str) -> int:
        """
        Args:
            image_url: Image URL (can be data URI with base64 or regular URL)
            
        Returns:
            Estimated token count for the image
        """
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Extract base64 data if it's a data URI
            if image_url.startswith("data:image"):
                # Format: data:image/jpeg;base64,<base64_data>
                base64_data = image_url.split(",", 1)[1] if "," in image_url else ""
            elif image_url.startswith("http://") or image_url.startswith("https://"):
                print(f"URL image: {image_url}, use conservative estimate for URL images")
                return 1024  # Conservative estimate for URL images
            else:
                # Assume it's already base64
                base64_data = image_url
            
            # Decode base64 to get image
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))
            # Get original dimensions
            width, height = image.size
            
            width_token = width // 28
            height_token = height // 28
            total_tokens = width_token * height_token
            return total_tokens
            
        except Exception as e:
            # If calculation fails, use conservative estimate
            self.task_log.log_step(
                "warning",
                "LLM | Image Token Estimation",
                f"Failed to calculate image tokens, using estimate: {str(e)}",
            )
            # Conservative estimate: assume medium-sized image
            return 1024

    def ensure_summary_context(
        self, message_history: list, summary_prompt: str
    ) -> tuple[bool, list]:
        """
        Check if current message_history + summary_prompt will exceed context
        If it will exceed, remove the last assistant-user pair and return False
        Return True to continue, False if messages have been rolled back
        """
        # Get token usage from the last LLM call
        last_prompt_tokens = self.last_call_tokens.get("prompt_tokens", 0)
        last_completion_tokens = self.last_call_tokens.get("completion_tokens", 0)
        buffer_factor = 1.5

        # Calculate token count for summary prompt
        summary_tokens = int(self._estimate_tokens(summary_prompt) * buffer_factor)

        # Calculate token count for the last user message in message_history
        last_user_tokens = 0
        if message_history[-1]["role"] in ("user", "tool"):
            content = message_history[-1]["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        last_user_tokens += self._estimate_image_tokens(item.get("image_url")['url'])
                        print(f"image estimated token: {last_user_tokens}")
                    elif item.get("type") == "text":
                        last_user_tokens += int(self._estimate_tokens(str(item.get("text"))) * buffer_factor)
                        print(f"text estimated token: {last_user_tokens}")
                    else:
                        logger.error(f"Unknown content type: {item.get('type')}")
            else:
                last_user_tokens = int(self._estimate_tokens(str(content)) * buffer_factor)

        # Calculate total token count: last prompt + completion + last user message + summary + reserved response space
        estimated_total = (
            last_prompt_tokens
            + last_completion_tokens
            + last_user_tokens
            + summary_tokens
            + self.max_tokens
            + 1000  # Add 1000 tokens as buffer
        )
        # print(f"{last_prompt_tokens=}")
        # print(f"{last_completion_tokens=}")
        # print(f"{last_user_tokens=}")
        # print(f"{summary_tokens=}")
        # print(f"{self.max_tokens=}")
        # print(f"{estimated_total=}")
        # print(f"{self.max_context_length=}")
        
        # # Print last_user_tokens content
        # if message_history and message_history[-1]["role"] == "user":
        #     last_user_content = message_history[-1]["content"]
        #     print(f"\n=== last_user_tokens content (length: {len(str(last_user_content))} chars) ===")
        #     print(f"{last_user_content}")
        #     print("=" * 80)
        
        # Print last_prompt_tokens content (reconstruct the prompt from last call)
        # Get system_prompt from task_log
        if hasattr(self, 'task_log') and self.task_log:
            main_agent_msg = getattr(self.task_log, 'main_agent_message_history', None)
            system_prompt = ''
            if isinstance(main_agent_msg, dict):
                system_prompt = main_agent_msg.get('system_prompt', '')
            
            # Reconstruct the message history from last call (remove the last user message which is tool result)
            last_call_message_history = message_history[:-1] if message_history and message_history[-1]["role"] in ("user", "tool") else message_history
            
            # Build the full prompt that was sent in the last call
            if system_prompt and last_call_message_history:
                # Reconstruct messages as they were sent to LLM
                full_prompt_parts = []
                full_prompt_parts.append(f"System Prompt:\n{system_prompt}\n")
                full_prompt_parts.append("\nMessage History:\n")
                for msg in last_call_message_history:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        # Handle multi-modal content
                        content_str = "\n".join([str(item) for item in content])
                    else:
                        content_str = str(content)
                    full_prompt_parts.append(f"[{role}]: {content_str}\n")
                
                # full_prompt = "\n".join(full_prompt_parts)
                # print(f"\n=== last_prompt_tokens content (length: {len(full_prompt)} chars, estimated tokens: {last_prompt_tokens}) ===")
                # print(f"{full_prompt[-5000:]}")  # Print first 5000 chars to avoid too long output
                # if len(full_prompt) > 5000:
                #     print(f"\n... (truncated, total length: {len(full_prompt)} chars)")
                # print("=" * 80)

        if estimated_total >= self.max_context_length:
            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                "Context limit reached, proceeding to step back and summarize the conversation",
            )

            # Remove all trailing tool/user messages (tool call results)
            while message_history and message_history[-1]["role"] in ("user", "tool"):
                message_history.pop()

            # Remove the assistant message with tool_calls that preceded them
            if message_history and message_history[-1]["role"] == "assistant":
                message_history.pop()

            self.task_log.log_step(
                "info",
                "LLM | Context Limit Reached",
                f"Removed the last assistant-user pair, current message_history length: {len(message_history)}",
            )

            return False, message_history

        self.task_log.log_step(
            "info",
            "LLM | Context Limit Not Reached",
            f"{estimated_total}/{self.max_context_length}",
        )
        return True, message_history

    def format_token_usage_summary(self) -> tuple[List[str], str]:
        """Format token usage statistics, return summary_lines for format_final_summary and log string"""
        token_usage = self.get_token_usage()

        total_input = token_usage.get("total_input_tokens", 0)
        total_output = token_usage.get("total_output_tokens", 0)
        cache_input = token_usage.get("total_cache_input_tokens", 0)

        summary_lines = []
        summary_lines.append("\n" + "-" * 20 + " Token Usage " + "-" * 20)
        summary_lines.append(f"Total Input Tokens: {total_input}")
        summary_lines.append(f"Total Cache Input Tokens: {cache_input}")
        summary_lines.append(f"Total Output Tokens: {total_output}")
        summary_lines.append("-" * (40 + len(" Token Usage ")))
        summary_lines.append("Pricing is disabled - no cost information available")
        summary_lines.append("-" * (40 + len(" Token Usage ")))

        # Generate log string
        log_string = (
            f"[{self.model_name}] Total Input: {total_input}, "
            f"Cache Input: {cache_input}, "
            f"Output: {total_output}"
        )

        return summary_lines, log_string

    def get_token_usage(self):
        return self.token_usage.copy()
