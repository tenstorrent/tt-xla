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

for tool in unzip zip strip sha256sum; do
	if ! command -v "$tool" >/dev/null 2>&1; then
		echo "Error: required tool '$tool' not found in PATH." >&2
		exit 1
	fi
done

relpath() {
	local target="$1"
	local base="$2"
	if command -v realpath >/dev/null 2>&1; then
		if rel=$(realpath --relative-to="$base" "$target" 2>/dev/null); then
			printf '%s\n' "$rel"
			return
		fi
	fi
	if command -v python3 >/dev/null 2>&1; then
		python3 -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$target" "$base"
		return
	fi
	echo "Error: unable to compute relative path; install GNU realpath or python3." >&2
	exit 1
}

deduplicate_files() {
	declare -A seen_hashes=()
	local total_dupes=0
	local total_dupe_bytes=0
	while IFS= read -r -d '' file; do
		# Skip any symlinks introduced earlier in the process.
		if [ -h "$file" ]; then
			continue
		fi
		local checksum
		checksum=$(sha256sum "$file" | awk '{print $1}')
		local rel_path=${file#"$temp_dir"/}
		if [[ -n ${seen_hashes[$checksum]+_} ]]; then
			local target_rel=${seen_hashes[$checksum]}
			local target_path="$temp_dir/$target_rel"
			local file_size
			file_size=$(stat -c%s "$file")
			local link_dir
			link_dir=$(dirname "$file")
			local relative_target
			relative_target=$(relpath "$target_path" "$link_dir")
			rm -f "$file"
			ln -s "$relative_target" "$file"
			total_dupes=$((total_dupes + 1))
			total_dupe_bytes=$((total_dupe_bytes + file_size))
			printf 'link %s -> %s: -%d bytes\n' "$rel_path" "$relative_target" "$file_size" >&2
		else
			seen_hashes["$checksum"]="$rel_path"
		fi
		done < <(find "$temp_dir" -type f -print0 | sort -z)
	echo "$total_dupes" "$total_dupe_bytes"
}

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

# Remove static library files from pjrt_plugin_tt/lib and track bytes removed.
while IFS= read -r -d '' archive_file; do
    rel_path=${archive_file#"$temp_dir"/}
    file_size=$(stat -c%s "$archive_file")
    total_static_removed=$((total_static_removed + file_size))
    printf 'delete %s: -%d bytes\n' "$rel_path" "$file_size" >&2
    rm -f "$archive_file"
done < <(find "$temp_dir" -type f -path '*/pjrt_plugin_tt/lib/*.a' -print0)

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

read -r total_dupes total_dupe_bytes < <(deduplicate_files)

(
	cd "$temp_dir"
	zip -X -q -r --symlinks "$temp_wheel" .
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
if [ "${total_dupes}" -gt 0 ]; then
	echo "Replaced duplicate files with symlinks: ${total_dupes} files; ${total_dupe_bytes} bytes before dedupe." >&2
else
	echo "No duplicate files replaced with symlinks." >&2
fi
echo "Repacked wheel: reduced by ${size_delta} bytes${percent_suffix}; new size ${final_size} bytes." >&2

trap - EXIT
cleanup
