# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Task logging and structured output module.

This module provides:
- TaskLog: Main dataclass for tracking task execution state and history
- StepLog: Individual step logging with timestamps and metadata
- ColoredFormatter: Console output formatting with color-coded log levels
- Utility functions for time handling and logger configuration

All logs are persisted to JSON files for later analysis and debugging.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Import colorama for cross-platform colored output
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True, strip=False)

# This will be set to the configured logger instance
logger = None


def get_color_for_level(level: str) -> str:
    """Get color code based on log level for better visual distinction"""
    if level == "ERROR":
        return f"{Fore.RED}{Style.BRIGHT}"
    elif level == "WARNING":
        return f"{Fore.YELLOW}{Style.BRIGHT}"
    elif level == "INFO":
        return f"{Fore.GREEN}{Style.BRIGHT}"
    elif level == "DEBUG":
        return f"{Fore.CYAN}{Style.BRIGHT}"
    else:
        return f"{Fore.WHITE}{Style.BRIGHT}"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors for better developer visualization"""

    def format(self, record):
        # Get timestamp and format it
        timestamp = self.formatTime(record, self.datefmt)

        # Color the level name based on severity
        level_color = get_color_for_level(record.levelname)
        level_reset = Style.RESET_ALL

        # Color the logger name (miroflow_agent)
        name_color = f"{Fore.BLUE}{Style.BRIGHT}"
        name_reset = Style.RESET_ALL

        # Get the message as is (icons are already added in log_step)
        message = record.getMessage()

        # Format with selective coloring
        formatted = f"[{timestamp}][{name_color}{record.name}{name_reset}][{level_color}{record.levelname}{level_reset}] - {message}"

        return formatted


def bootstrap_logger() -> logging.Logger:
    """Configure the miroflow_agent logger with consistent formatting"""

    global logger

    # Configure miroflow_agent logger
    miroflow_agent_logger = logging.getLogger("miroflow_agent")

    # Check if logger already has handlers to prevent duplicate configuration
    if miroflow_agent_logger.handlers:
        logger = miroflow_agent_logger
        return miroflow_agent_logger

    # Create formatter with consistent format
    formatter = ColoredFormatter(
        "%(asctime)s,%(msecs)03d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add our handler with the specified formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    miroflow_agent_logger.addHandler(handler)
    miroflow_agent_logger.setLevel(logging.DEBUG)

    # Disable propagation to prevent duplicate logging from root logger
    miroflow_agent_logger.propagate = False

    # Set the global logger variable
    logger = miroflow_agent_logger

    return miroflow_agent_logger


def get_utc_plus_8_time() -> str:
    """Get UTC+8 timezone current time string"""
    utc_plus_8 = timezone(timedelta(hours=8))
    return datetime.now(utc_plus_8).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LLMCallLog:
    """Record technical details of LLM calls"""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    error: Optional[str] = None


@dataclass
class ToolCallLog:
    """Record detailed information of tool calls"""

    server_name: str
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    call_time: Optional[str] = None


@dataclass
class StepLog:
    """Record detailed information of task execution steps"""

    step_name: str
    message: str
    timestamp: str
    info_level: Literal["info", "warning", "error", "debug"] = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate info_level after initialization"""
        valid_levels = {"info", "warning", "error", "debug"}
        if self.info_level not in valid_levels:
            raise ValueError(
                f"info_level must be one of {valid_levels}, got '{self.info_level}'"
            )


@dataclass
class TaskLog:
    status: str = "running"
    start_time: str = ""
    end_time: str = ""

    task_id: str = ""
    input: Any = None
    ground_truth: str = ""
    final_boxed_answer: str = ""
    final_judge_result: str = ""
    judge_type: str = ""
    eval_details: Optional[Dict[str, Any]] = None  # For DeepSearchQA metrics
    error: str = ""

    # Main records: main agent conversation turns
    current_main_turn_id: int = 0
    current_sub_agent_turn_id: int = 0
    sub_agent_counter: int = 0
    current_sub_agent_session_id: Optional[str] = None

    env_info: Optional[dict] = field(default_factory=dict)
    log_dir: str = "logs"

    main_agent_message_history: List[Dict[str, Any]] = field(default_factory=list)
    sub_agent_message_history_sessions: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )

    step_logs: List[StepLog] = field(default_factory=list)
    trace_data: Dict[str, Any] = field(default_factory=dict)

    # Track images data for saving to separate file
    _images_data: Dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def start_sub_agent_session(
        self, sub_agent_name: str, subtask_description: str
    ) -> str:
        """Start a new sub-agent session"""
        self.sub_agent_counter += 1
        session_id = f"{sub_agent_name}_{self.sub_agent_counter}"
        self.current_sub_agent_session_id = session_id

        # Record sub-agent session start
        self.log_step(
            "info",
            f"{sub_agent_name} | Session Start",
            f"Starting {session_id} for subtask: {subtask_description[:100]}{'...' if len(subtask_description) > 100 else ''}",
            metadata={"session_id": session_id, "subtask": subtask_description},
        )

        return session_id

    def end_sub_agent_session(self, sub_agent_name: str) -> Optional[str]:
        """End the current sub-agent session"""
        self.log_step(
            "info",
            f"{sub_agent_name} | Session End",
            f"Ending {self.current_sub_agent_session_id}",
            metadata={"session_id": self.current_sub_agent_session_id},
        )
        self.current_sub_agent_session_id = None
        return None

    def log_step(
        self,
        info_level: Literal["info", "warning", "error", "debug"],
        step_name: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record execution step"""
        # Add icons to step_name based on content
        icon = ""
        if "Tool Call Start" in step_name:
            icon = "▶️ "
        elif "Tool Call Success" in step_name:
            icon = "✅ "
        elif "Tool Call Error" in step_name or (
            "error" in info_level and "tool" in step_name.lower()
        ):
            icon = "❌ "
        elif "agent-" in step_name:
            icon = "🤖 "
        elif "Main Agent" in step_name:
            icon = "👑 "
        elif "LLM" in step_name:
            icon = "🧠 "
        elif "ToolManager" in step_name or "Tool Call" in step_name:
            icon = "🔧 "
        elif "tool-python" in step_name.lower():
            icon = "🐍 "
        elif "tool-google-search" in step_name.lower():
            icon = "🔍 "
        elif "tool-browser" in step_name.lower() or "playwright" in step_name.lower():
            icon = "🌐 "

        # Add icon to step_name
        step_name_with_icon = f"{icon}{step_name}"

        step_log = StepLog(
            step_name=step_name_with_icon,
            message=message,
            timestamp=get_utc_plus_8_time(),
            info_level=info_level,
            metadata=metadata or {},
        )

        self.step_logs.append(step_log)

        # Print the structured log to console using the configured logger
        log_message = f"{step_name_with_icon}: {message}"

        # Ensure logger is configured
        global logger
        if logger is None:
            logger = bootstrap_logger()

        if info_level == "error":
            logger.error(log_message)
        elif info_level == "warning":
            logger.warning(log_message)
        elif info_level == "debug":
            logger.debug(log_message)
        else:  # info
            logger.info(log_message)

    def serialize_for_json(self, obj, image_key_prefix: str = ""):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            # Check if this is a message with content that might contain images
            if "role" in obj and "content" in obj:
                return self._serialize_message(obj, image_key_prefix)
            return {k: self.serialize_for_json(v, image_key_prefix) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.serialize_for_json(item, image_key_prefix) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self.serialize_for_json(obj.__dict__, image_key_prefix)
        else:
            return obj

    def _serialize_message(self, message: dict, image_key_prefix: str = "") -> dict:
        """
        Serialize a message, replacing base64 image data with placeholder.

        Args:
            message: A message dictionary with 'role' and 'content' fields
            image_key_prefix: Prefix for image keys (e.g., "main_0", "sub_1")

        Returns:
            Serialized message with base64 data replaced by "<image>"
        """
        result = {"role": message["role"]}
        content = message["content"]

        # Handle different content formats
        if isinstance(content, list):
            # Multi-modal content (list of content items)
            serialized_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # Extract base64 data before replacing
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url and image_url != "<image>" and image_url.startswith("data:image"):
                            # Generate unique key using global counter
                            image_key = f"{image_key_prefix}_{len(self._images_data)}" if image_key_prefix else f"image_{len(self._images_data)}"
                            # Store the base64 data
                            self._images_data[image_key] = image_url

                        # Replace with placeholder (use a copy to avoid modifying original)
                        serialized_content.append({
                            "type": "image_url",
                            "image_url": {"url": "<image>"}
                        })
                    elif item.get("type") == "text":
                        # Keep text as is (make a copy to avoid modifying original)
                        serialized_content.append(item.copy())
                    else:
                        # Recursively serialize other types
                        serialized_content.append(self.serialize_for_json(item))
                else:
                    serialized_content.append(self.serialize_for_json(item))
            result["content"] = serialized_content
        elif isinstance(content, str):
            # Text-only content - check if it contains base64 data
            if content.startswith("data:image") and ";base64," in content:
                # Store the base64 data
                image_key = f"{image_key_prefix}_{len(self._images_data)}" if image_key_prefix else f"image_{len(self._images_data)}"
                self._images_data[image_key] = content
                result["content"] = "<image>"
            else:
                result["content"] = content
        else:
            # Other formats - serialize normally
            result["content"] = self.serialize_for_json(content)

        # Copy any other fields
        for key, value in message.items():
            if key not in ("role", "content"):
                result[key] = self.serialize_for_json(value)

        return result

    def to_json(self) -> str:
        """
        Serialize the TaskLog to a JSON string.

        Converts the dataclass to a dictionary, handles non-JSON-serializable
        objects (like Path), and returns a formatted JSON string.

        Returns:
            A JSON string representation of the task log with 2-space indentation.

        Note:
            Falls back to ASCII encoding if Unicode encoding fails.
        """
        # Clear _images_data before calling asdict() to prevent base64 data in JSON
        # Note: _images_data will be repopulated during serialize_for_json()
        self._images_data.clear()

        # Convert to dict first (without _images_data)
        data_dict = asdict(self)

        # Serialize any non-JSON-serializable objects
        # Use different prefixes for main and sub agents
        serialized_dict = self.serialize_for_json(data_dict, image_key_prefix="main")
        try:
            return json.dumps(serialized_dict, ensure_ascii=False, indent=2)
        except UnicodeEncodeError as e:
            # Fallback: try with ASCII encoding if Unicode fails
            print(f"Warning: Unicode encoding failed, falling back to ASCII: {e}")
            return json.dumps(serialized_dict, ensure_ascii=True, indent=2)

    def _save_images_as_files(self, timestamp: str) -> List[str]:
        """
        Save images as individual files in a dedicated directory.

        Args:
            timestamp: Timestamp string for naming

        Returns:
            List of saved image file paths
        """
        if not self._images_data:
            return []

        import base64

        # Create images directory under save_images subdirectory
        images_dir = f"{self.log_dir}/save_images/task_{self.task_id}_{timestamp}_images"
        os.makedirs(images_dir, exist_ok=True)

        saved_paths = []

        for image_key, base64_data in self._images_data.items():
            try:
                # Extract MIME type and file extension
                if base64_data.startswith("data:image/"):
                    # Parse data URL (e.g., "data:image/jpeg;base64,...")
                    mime_part = base64_data.split(";")[0]
                    mime_type = mime_part.replace("data:", "")

                    # Map MIME type to extension
                    mime_to_ext = {
                        "image/jpeg": ".jpg",
                        "image/jpg": ".jpg",
                        "image/png": ".png",
                        "image/gif": ".gif",
                        "image/webp": ".webp",
                        "image/bmp": ".bmp",
                    }
                    ext = mime_to_ext.get(mime_type, ".jpg")

                    # Remove data URL prefix
                    if "," in base64_data:
                        base64_data = base64_data.split(",", 1)[1]
                else:
                    # Default to JPEG if no data URL prefix
                    ext = ".jpg"

                # Decode base64
                image_bytes = base64.b64decode(base64_data)

                # Generate filename based on image key
                # image_key format: "main_0", "main_1", etc. or "image_0", "image_1"
                safe_image_key = image_key.replace("/", "_").replace("\\", "_")
                image_filename = f"{self.task_id}_{safe_image_key}{ext}"
                image_path = os.path.join(images_dir, image_filename)

                # Save image file
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                saved_paths.append(image_path)

            except Exception as e:
                print(f"Warning: Failed to save image {image_key}: {e}")

        if saved_paths:
            print(f"Saved {len(saved_paths)} image(s) to directory: {images_dir}")

        return saved_paths

    def save(self):
        """Save as a single JSON file"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = (
            self.start_time.replace(":", "-").replace(".", "-").replace(" ", "-")
        )

        filename = f"{self.log_dir}/task_{self.task_id}_{timestamp}.json"

        # This will populate self._images_data during serialization
        json_content = self.to_json()

        # Save the main JSON log file
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json_content)
        except UnicodeEncodeError as e:
            # Fallback: try with different encoding if UTF-8 fails
            print(f"Warning: UTF-8 encoding failed, trying with system default: {e}")
            with open(filename, "w") as f:
                f.write(json_content)

        # Save images as files if any images were found
        saved_image_paths = self._save_images_as_files(timestamp)

        # Clear _images_data after saving to prevent accumulation across multiple save() calls
        self._images_data.clear()

        return filename

    @classmethod
    def from_dict(cls, d: dict) -> "TaskLog":
        """
        Create a TaskLog instance from a dictionary.

        Args:
            d: Dictionary containing TaskLog field values.

        Returns:
            A new TaskLog instance initialized with the dictionary values.

        Note:
            The dictionary keys should match the TaskLog field names.
        """
        return cls(**d)
