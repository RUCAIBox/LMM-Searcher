# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def get_successful_log_paths(jsonl_file_path: str) -> list:
    """
    Collects the paths of successful log files from a dataset.

    This function extracts log file paths of successful records based on
    the value of `final_judge_result`. It scans both the JSONL file and
    all JSON files in the directory to ensure all successful cases are captured.

    Success is determined by:
    - `PASS_AT_K_SUCCESS` for records in JSONL files
    - `CORRECT` for records in individual JSON files

    Args:
        jsonl_file_path (str): Path to a JSONL file or a directory of JSON files.

    Returns:
        list: A list of log file paths for successful records.
    """
    log_paths = []
    seen_paths = set()  # To avoid duplicates

    if jsonl_file_path.endswith(".jsonl"):
        # First, extract paths from the JSONL file
        jsonl_dir = os.path.abspath(os.path.dirname(jsonl_file_path))

        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("final_judge_result") == "PASS_AT_K_SUCCESS":
                            log_path = data.get("log_file_path")
                            if log_path:
                                # Resolve relative paths
                                if not os.path.isabs(log_path):
                                    log_path = os.path.join(jsonl_dir, log_path)
                                    log_path = os.path.abspath(log_path)

                                # Verify the file actually exists and is CORRECT
                                # (PASS_AT_K_SUCCESS may point to an attempt that's not actually CORRECT)
                                if os.path.exists(log_path) and log_path not in seen_paths:
                                    try:
                                        with open(log_path, "r", encoding="utf-8") as f:
                                            file_data = json.load(f)
                                        # Only include if the file itself is marked as CORRECT
                                        if file_data.get("final_judge_result") == "CORRECT":
                                            log_paths.append(log_path)
                                            seen_paths.add(log_path)
                                    except Exception:
                                        # If we can't read the file, skip it
                                        continue
                    except json.JSONDecodeError:
                        continue

        # Then, scan all JSON files in the directory to find CORRECT cases
        # This captures successful tasks that may not be in the JSONL summary
        for filename in os.listdir(jsonl_dir):
            if not filename.endswith(".json") or filename.endswith("_images.json"):
                continue

            filepath = os.path.join(jsonl_dir, filename)
            abs_filepath = os.path.abspath(filepath)

            # Skip if already processed
            if abs_filepath in seen_paths:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            try:
                final_judge_result = data.get("final_judge_result")
                if final_judge_result == "CORRECT":
                    log_paths.append(abs_filepath)
                    seen_paths.add(abs_filepath)
            except KeyError:
                continue

    else:
        # If directory path is provided directly
        filenames = os.listdir(jsonl_file_path)
        filenames = [filename for filename in filenames if filename.endswith(".json")]
        for filename in filenames:
            filepath = os.path.join(jsonl_file_path, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            try:
                final_judge_result = data["final_judge_result"]
            except KeyError:
                print(data.keys())
                continue
            if final_judge_result == "CORRECT":
                log_paths.append(filepath)

    return log_paths


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract successful log paths from JSONL file"
    )
    parser.add_argument(
        "file_path", help="Path to the JSONL file containing benchmark results"
    )
    parser.add_argument(
        "--tool-defs",
        help="Path to a JSON file containing tool_definitions (generated by dump_tool_definitions.py). "
             "Used to rebuild system prompts for old data collected with use_tool_calls=True.",
        default=None,
    )
    args = parser.parse_args()

    result = get_successful_log_paths(args.file_path)

    # Get the parent directory of args.file_path
    parent_dir = os.path.abspath(os.path.dirname(args.file_path))

    # Create successful logs directory
    success_log_dir = parent_dir + "/successful_logs"
    success_chatml_log_dir = parent_dir + "/successful_chatml_logs"
    success_sharegpt_log_dir = parent_dir + "/successful_sharegpt_logs"
    os.makedirs(success_log_dir, exist_ok=True)
    os.makedirs(success_sharegpt_log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {len(result)} successful logs")
    print(f"{'='*60}")
    print(f"Output directories:")
    print(f"  • Logs:      {success_log_dir}")
    print(f"  • ShareGPT:  {success_sharegpt_log_dir}")

    # Copy files with progress indicator
    copied_files = 0
    copied_images = 0
    skipped_images = 0

    for i, path in enumerate(result, 1):
        basename = os.path.basename(path)
        shutil.copy(path, f"{success_log_dir}/{basename}")
        copied_files += 1

        # Get the base filename without extension
        file_basename = os.path.splitext(basename)[0]

        # Try to copy the corresponding images directory
        # First try new format: save_images/task_xxx_images
        path_dir = os.path.dirname(path)
        new_format_images_dir = os.path.join(path_dir, "save_images", f"{file_basename}_images")

        # Then try old format: task_xxx_images (in parent dir)
        old_format_images_dir = path.replace(".json", "_images")

        # Determine which format exists and copy it
        if os.path.exists(new_format_images_dir) and os.path.isdir(new_format_images_dir):
            images_basename = os.path.basename(new_format_images_dir)
            dest_images_dir = os.path.join(success_log_dir, images_basename)
            if os.path.exists(dest_images_dir):
                shutil.rmtree(dest_images_dir)
            shutil.copytree(new_format_images_dir, dest_images_dir)
            copied_images += 1
        elif os.path.exists(old_format_images_dir) and os.path.isdir(old_format_images_dir):
            images_basename = os.path.basename(old_format_images_dir)
            dest_images_dir = f"{success_log_dir}/{images_basename}"
            if os.path.exists(dest_images_dir):
                shutil.rmtree(dest_images_dir)
            shutil.copytree(old_format_images_dir, dest_images_dir)
            copied_images += 1
        else:
            skipped_images += 1

        # Also copy the old-style images JSON file if it exists (for backward compatibility)
        images_file = path.replace(".json", "_images.json")
        if os.path.exists(images_file):
            images_basename = os.path.basename(images_file)
            shutil.copy(images_file, f"{success_log_dir}/{images_basename}")

        # Simple progress indicator
        if i % 10 == 0 or i == len(result):
            print(f"  Progress: [{i}/{len(result)}] files copied...", end='\r')

    print(f"\n✓ Copied {copied_files} log files")
    print(f"  • With images: {copied_images}")
    print(f"  • Without images: {skipped_images}")

    # # Convert to ChatML format (currently disabled)
    # # Uncomment if ChatML format is needed
    # print(f"\n[1/3] Converting to ChatML format...")
    # result = subprocess.run(
    #     f"uv run utils/converters/convert_to_chatml_auto_batch.py {success_log_dir}/*.json -o {success_chatml_log_dir}",
    #     shell=True,
    #     capture_output=True,
    #     text=True
    # )
    # if result.returncode == 0:
    #     print(f"  ✓ ChatML conversion completed")
    # else:
    #     print(f"  ⚠ ChatML conversion had warnings")
    #
    # result = subprocess.run(
    #     f"uv run utils/merge_chatml_msgs_to_one_json.py --input_dir {success_chatml_log_dir}",
    #     shell=True,
    #     capture_output=True,
    #     text=True
    # )
    # if result.returncode == 0:
    #     print(f"  ✓ ChatML messages merged")
    # else:
    #     print(f"  ⚠ ChatML merge had warnings")

    # Convert to ShareGPT format (with multiprocessing)
    print(f"\n[1/2] Converting to ShareGPT format (parallel)...")

    sharegpt_files = [f for f in os.listdir(success_log_dir) if f.endswith(".json") and not f.endswith("_images.json")]

    # Determine number of worker processes (use CPU count, but cap at 16 for safety)
    num_workers = min(cpu_count(), 64)
    print(f"  Using {num_workers} parallel workers for {len(sharegpt_files)} files...")

    tool_defs_path = args.tool_defs

    def convert_single_file(json_file):
        """Convert a single file to ShareGPT format"""
        json_path = os.path.join(success_log_dir, json_file)
        try:
            cmd = ["uv", "run", "utils/converters/convert_to_sharegpt.py", json_path, success_sharegpt_log_dir]
            if tool_defs_path:
                cmd.extend(["--tool-defs", tool_defs_path])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout per file
            )
            if result.returncode == 0:
                return (json_file, True, None)
            else:
                return (json_file, False, result.stderr)
        except subprocess.TimeoutExpired:
            return (json_file, False, "Timeout after 60 seconds")
        except Exception as e:
            return (json_file, False, str(e))

    # Process files in parallel
    converted_count = 0
    failed_count = 0
    failed_files = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(convert_single_file, json_file): json_file
                          for json_file in sharegpt_files}

        # Process completed tasks with progress indicator
        completed = 0
        for future in as_completed(future_to_file):
            completed += 1
            json_file, success, error = future.result()

            if success:
                converted_count += 1
            else:
                failed_count += 1
                failed_files.append((json_file, error))

            # Progress indicator
            if completed % 10 == 0 or completed == len(sharegpt_files):
                print(f"  Progress: [{completed}/{len(sharegpt_files)}] files processed...", end='\r')

    print(f"\n  ✓ Converted: {converted_count} files")
    if failed_count > 0:
        print(f"  ⚠ Failed: {failed_count} files")
        # Show first 5 failures only
        for json_file, error in failed_files[:5]:
            print(f"    - {json_file}: {error[:50]}...")
        if failed_count > 5:
            print(f"    ... and {failed_count - 5} more failures")

    # Merge all ShareGPT logs into one file
    print(f"\n[2/2] Merging ShareGPT logs...")
    merged_data = []
    for json_file in os.listdir(success_sharegpt_log_dir):
        if json_file.endswith("_sharegpt.json"):
            json_path = os.path.join(success_sharegpt_log_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.append(data)
            except Exception as e:
                print(f"  ⚠ Failed to read {json_file}: {e}")

    # Save merged file
    merged_file = os.path.join(success_sharegpt_log_dir, "merged.json")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Merged {len(merged_data)} ShareGPT logs")
    print(f"  Output: {merged_file}")

    # Generate parquet file with base64 encoded images
    print(f"\n[3/3] Generating parquet file with base64 images...")
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        def process_sample(data_with_index):
            """Process a single sample for parquet format"""
            data, index = data_with_index
            # Handle both 'conversations' and 'messages' keys
            conversations = data.get('conversations', data.get('messages', []))

            # Process images if present, always initialize with empty list
            images = []
            if "images" in data.keys() and data["images"]:
                for image_path in data['images']:
                    try:
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()
                        images.append(image_bytes)
                    except Exception as e:
                        print(f"Warning: Failed to read image {image_path}: {e}")
                        continue

            # Always return all three fields for consistency
            return {
                "id": data.get('id', f"sample_{index}"),
                "images": images,
                "conversations": conversations
            }

        # Process all samples
        processed_data = []
        for idx, data in enumerate(merged_data):
            processed = process_sample((data, idx))
            processed_data.append(processed)

        # Save as parquet
        parquet_file = os.path.join(success_sharegpt_log_dir, "merged.parquet")
        data_table = pa.Table.from_pylist(processed_data)
        pq.write_table(
            data_table,
            parquet_file,
            row_group_size=1000,
            compression="NONE",
            use_dictionary=False,
            write_batch_size=1,
            write_page_index=True,
        )

        samples_with_images = sum(1 for d in processed_data if d['images'])
        print(f"  ✓ Generated parquet file")
        print(f"    Total samples: {len(processed_data)}")
        print(f"    With images: {samples_with_images}")
        print(f"    Output: {parquet_file}")

    except ImportError:
        print(f"  ⚠ Skipped parquet generation (missing pandas/pyarrow)")
        print(f"    Install with: uv add pandas pyarrow")
    except Exception as e:
        print(f"  ⚠ Failed to generate parquet: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"✓ Processing completed!")
    print(f"{'='*60}")
    print(f"Generated files:")
    print(f"  1. {success_sharegpt_log_dir}/merged.json")
    print(f"  2. {success_sharegpt_log_dir}/merged.parquet")
    print(f"{'='*60}\n")
