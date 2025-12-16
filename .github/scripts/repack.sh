#!/usr/bin/env bash

# Repackage an existing Python wheel to reduce size by pruning static
# archives and stripping shared libraries before recreating the wheel.

set -euo pipefail

usage() {
	echo "Usage: $0 path/to/wheel.whl" >&2
	exit 1
}

if [ $# -ne 1 ]; then
	usage
fi

wheel_input="$1"

if [ ! -f "$wheel_input" ]; then
	echo "Error: wheel file '$wheel_input' not found." >&2
	exit 1
fi

if command -v realpath >/dev/null 2>&1; then
	wheel_path=$(realpath "$wheel_input")
else
	wheel_dir_tmp=$(cd "$(dirname "$wheel_input")" && pwd)
	wheel_path="$wheel_dir_tmp/$(basename "$wheel_input")"
fi

for tool in unzip zip strip; do
	if ! command -v "$tool" >/dev/null 2>&1; then
		echo "Error: required tool '$tool' not found in PATH." >&2
		exit 1
	fi
done

temp_dir=$(mktemp -d)
wheel_dir=$(dirname "$wheel_path")
temp_wheel=$(mktemp --tmpdir="$wheel_dir" tmpwheel.XXXXXX.whl)
rm -f "$temp_wheel"
original_size=$(stat -c%s "$wheel_path")
total_static_removed=0

cleanup() {
	rm -rf "$temp_dir"
	rm -f "$temp_wheel"
}

trap cleanup EXIT

unzip -q "$wheel_path" -d "$temp_dir"

# Remove all static library files (*.a) and track total bytes removed.
while IFS= read -r -d '' archive_file; do
	file_size=$(stat -c%s "$archive_file")
	total_static_removed=$((total_static_removed + file_size))
	printf 'delete %s: -%d bytes\n' "${archive_file#"$temp_dir"/}" "$file_size" >&2
done < <(find "$temp_dir" -type f -name '*.a' -print0)

find "$temp_dir" -type f -name '*.a' -delete

# Strip shared libraries and report savings per file.
while IFS= read -r -d '' so_file; do
	before_size=$(stat -c%s "$so_file")
	strip --strip-unneeded "$so_file"
	after_size=$(stat -c%s "$so_file")
	size_saved=$((before_size - after_size))
	rel_path=${so_file#"$temp_dir"/}
	percent_note=""
	if [ "$size_saved" -gt 0 ] && [ "$before_size" -gt 0 ]; then
		percent_saved=$(awk -v orig="$before_size" -v delta="$size_saved" 'BEGIN { printf "%.2f", (delta / orig) * 100 }')
		percent_note=" (${percent_saved}%)"
	fi
	printf 'strip %s: -%d bytes%s\n' "$rel_path" "$size_saved" "$percent_note" >&2
done < <(find "$temp_dir" -type f -name '*.so' -print0)

(
	cd "$temp_dir"
	zip -X -q -r "$temp_wheel" .
)

mv -f "$temp_wheel" "$wheel_path"

final_size=$(stat -c%s "$wheel_path")
size_delta=$((original_size - final_size))

percent_suffix=""
if [ "$size_delta" -gt 0 ] && [ "$original_size" -gt 0 ]; then
	percent_saved=$(awk -v orig="$original_size" -v delta="$size_delta" 'BEGIN { printf "%.2f", (delta / orig) * 100 }')
	percent_suffix=" (${percent_saved}%)"
fi

echo "Removed static archives: ${total_static_removed} bytes." >&2
echo "Repacked wheel: reduced by ${size_delta} bytes${percent_suffix}; new size ${final_size} bytes." >&2

trap - EXIT
cleanup
