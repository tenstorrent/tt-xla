#!/usr/bin/env python3
"""
Script to log folder sizes for monitoring disk usage during tests.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def get_folder_size(path: Path) -> int:
    """Get the size of a folder in bytes using du command."""
    try:
        result = subprocess.run(
            ["du", "-sb", str(path)],
            capture_output=True,
            text=True,
            check=True
        )
        # du -sb returns size in bytes followed by path
        size_str = result.stdout.split()[0]
        return int(size_str)
    except (subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_disk_usage() -> Dict[str, any]:
    """Get overall disk usage statistics."""
    try:
        result = subprocess.run(
            ["df", "-B1", "/"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 4:
                return {
                    "total": int(parts[1]),
                    "used": int(parts[2]),
                    "available": int(parts[3]),
                    "percent": parts[4] if len(parts) > 4 else "N/A"
                }
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return {"total": 0, "used": 0, "available": 0, "percent": "N/A"}


def collect_folder_sizes(test_name: Optional[str] = None) -> Dict[str, any]:
    """Collect sizes of important folders."""
    folders_to_monitor = [
        # Cache directories
        Path.home() / ".cache",
        Path("/mnt/dockercache"),
        Path("/tmp"),
        Path("/var/tmp"),

        # Project directories
        Path("/__w/tt-xla/tt-xla"),
        Path("/__w/tt-xla/tt-xla/build"),
        Path("/__w/tt-xla/tt-xla/tests"),
        Path("/__w/tt-xla/tt-xla/.pytest_cache"),

        # Python package directories
        Path("/__w/tt-xla/tt-xla/python_package"),
        Path("/__w/tt-xla/tt-xla/venv"),

        # Test artifact directories
        Path("/__w/tt-xla/tt-xla/results"),
        Path("/__w/tt-xla/tt-xla/output"),
        Path("/__w/tt-xla/tt-xla/outputs"),
        Path("/__w/tt-xla/tt-xla/artifacts"),
        Path("/__w/tt-xla/tt-xla/orbax"),
        Path("/__w/tt-xla/tt-xla/model_outputs"),
        Path("/__w/tt-xla/tt-xla/checkpoints"),

        # User-specific cache
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "jax",
        Path.home() / ".cache" / "jaxlib",
        Path.home() / ".cache" / "lfcache",
        Path.home() / ".cache" / "url_cache",
    ]

    sizes = {}
    total_size = 0

    for folder in folders_to_monitor:
        if folder.exists():
            size = get_folder_size(folder)
            sizes[str(folder)] = {
                "bytes": size,
                "human": format_size(size)
            }
            total_size += size

    # Get disk usage
    disk_usage = get_disk_usage()

    return {
        "timestamp": datetime.now().isoformat(),
        "test_name": test_name,
        "folder_sizes": sizes,
        "total_monitored": {
            "bytes": total_size,
            "human": format_size(total_size)
        },
        "disk_usage": {
            "total": format_size(disk_usage["total"]),
            "used": format_size(disk_usage["used"]),
            "available": format_size(disk_usage["available"]),
            "percent": disk_usage["percent"]
        }
    }


def find_large_files(base_path: Path, min_size_mb: int = 100, limit: int = 20) -> List[Dict[str, any]]:
    """Find large files in the specified path."""
    large_files = []
    try:
        # Use find command to locate large files
        min_size_bytes = min_size_mb * 1024 * 1024
        result = subprocess.run(
            ["find", str(base_path), "-type", "f", "-size", f"+{min_size_mb}M", "-exec", "ls", "-la", "{}", ";"],
            capture_output=True,
            text=True,
            timeout=30  # Timeout after 30 seconds
        )

        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[:limit]:
                if line:
                    parts = line.split()
                    if len(parts) >= 9:
                        try:
                            size = int(parts[4])
                            path = ' '.join(parts[8:])
                            large_files.append({
                                "path": path,
                                "size_bytes": size,
                                "size_human": format_size(size)
                            })
                        except (ValueError, IndexError):
                            pass
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Sort by size descending
    large_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    return large_files[:limit]


def main():
    """Main function to log folder sizes."""
    import argparse

    parser = argparse.ArgumentParser(description="Log folder sizes for CI monitoring")
    parser.add_argument("--test-name", help="Name of the test being run")
    parser.add_argument("--output", default="folder_sizes.json", help="Output JSON file")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--find-large-files", action="store_true", help="Also find large files")
    parser.add_argument("--min-file-size", type=int, default=100, help="Minimum file size in MB to report")

    args = parser.parse_args()

    # Collect folder sizes
    data = collect_folder_sizes(args.test_name)

    # Find large files if requested
    if args.find_large_files:
        workspace_path = Path("/__w/tt-xla/tt-xla")
        if workspace_path.exists():
            large_files = find_large_files(workspace_path, args.min_file_size)
            data["large_files"] = large_files

    # Handle output
    output_path = Path(args.output)

    if args.append and output_path.exists():
        # Load existing data
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (json.JSONDecodeError, IOError):
            existing_data = []

        # Append new data
        existing_data.append(data)
        output_data = existing_data
    else:
        output_data = [data]

    # Write output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Also print summary to stdout
    print(f"\n{'='*60}")
    print(f"Folder Size Report - {data['timestamp']}")
    if args.test_name:
        print(f"Test: {args.test_name}")
    print(f"{'='*60}")
    print(f"\nDisk Usage:")
    print(f"  Total: {data['disk_usage']['total']}")
    print(f"  Used: {data['disk_usage']['used']} ({data['disk_usage']['percent']})")
    print(f"  Available: {data['disk_usage']['available']}")
    print(f"\nTotal Monitored: {data['total_monitored']['human']}")
    print(f"\nTop 10 Largest Folders:")

    # Sort folders by size
    sorted_folders = sorted(
        data['folder_sizes'].items(),
        key=lambda x: x[1]['bytes'],
        reverse=True
    )[:10]

    for folder, info in sorted_folders:
        if info['bytes'] > 0:
            print(f"  {info['human']:>10} - {folder}")

    # Print large files if found
    if args.find_large_files and "large_files" in data and data["large_files"]:
        print(f"\nLarge Files (>{args.min_file_size}MB):")
        for file_info in data["large_files"][:10]:
            print(f"  {file_info['size_human']:>10} - {file_info['path']}")

    print(f"{'='*60}\n")

    # Write data to file
    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()