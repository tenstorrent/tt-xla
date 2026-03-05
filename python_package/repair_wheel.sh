#!/bin/bash
set -e

WHEEL=$1
DEST_DIR=$2

echo "Repairing wheel: $WHEEL"

# Create temp directory
TEMP_DIR=$(mktemp -d)
EXTRACT_DIR="$TEMP_DIR/wheel"
mkdir -p "$EXTRACT_DIR"

# Extract wheel
unzip -q "$WHEEL" -d "$EXTRACT_DIR"

# Find all .so files in pjrt_plugin_tt/lib and lib64
LIB_DIRS=("$EXTRACT_DIR/pjrt_plugin_tt/lib" "$EXTRACT_DIR/pjrt_plugin_tt/lib64")

# Collect system dependencies that need to be bundled
SYSTEM_LIBS=()
for lib_dir in "${LIB_DIRS[@]}"; do
    if [ -d "$lib_dir" ]; then
        echo "Analyzing libraries in $lib_dir"
        for so_file in "$lib_dir"/*.so*; do
            if [ -f "$so_file" ]; then
                echo "  Checking dependencies of $(basename $so_file)"
                # Use ldd to find dependencies
                while IFS= read -r line; do
                    # Extract library path from ldd output
                    if [[ $line =~ =\>\ ([^\ ]+) ]]; then
                        dep_path="${BASH_REMATCH[1]}"
                        dep_name=$(basename "$dep_path")

                        # Only bundle system libraries (not already in our wheel)
                        # Skip linux-vdso, ld-linux, libc, libm, libpthread, libdl, librt (they're in manylinux policy)
                        if [[ ! "$dep_name" =~ ^(linux-vdso|ld-linux|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so) ]] && \
                           [[ "$dep_path" == /lib* || "$dep_path" == /usr/lib* ]] && \
                           [ ! -f "$lib_dir/$dep_name" ]; then
                            SYSTEM_LIBS+=("$dep_path")
                        fi
                    fi
                done < <(ldd "$so_file" 2>/dev/null || true)
            fi
        done
    fi
done

# Copy system libraries into the wheel
if [ ${#SYSTEM_LIBS[@]} -gt 0 ]; then
    # Remove duplicates
    SYSTEM_LIBS=($(printf '%s\n' "${SYSTEM_LIBS[@]}" | sort -u))

    echo "Bundling system dependencies:"
    TARGET_LIB_DIR="$EXTRACT_DIR/pjrt_plugin_tt/lib64"
    mkdir -p "$TARGET_LIB_DIR"

    for sys_lib in "${SYSTEM_LIBS[@]}"; do
        if [ -f "$sys_lib" ]; then
            echo "  Copying $(basename $sys_lib)"
            cp "$sys_lib" "$TARGET_LIB_DIR/"
        fi
    done
fi

# Set RPATH for all .so files to find dependencies in lib/lib64
echo "Setting RPATH..."
for lib_dir in "${LIB_DIRS[@]}"; do
    if [ -d "$lib_dir" ]; then
        for so_file in "$lib_dir"/*.so*; do
            if [ -f "$so_file" ]; then
                patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64' "$so_file" 2>/dev/null || true
            fi
        done
    fi
done

# Also set RPATH for Python extension modules
for ext_module in "$EXTRACT_DIR"/**/*.so; do
    if [ -f "$ext_module" ] && [[ ! "$ext_module" =~ pjrt_plugin_tt/lib ]]; then
        echo "Setting RPATH for extension: $(basename $ext_module)"
        patchelf --set-rpath '$ORIGIN:$ORIGIN/pjrt_plugin_tt/lib:$ORIGIN/pjrt_plugin_tt/lib64' "$ext_module" 2>/dev/null || true
    fi
done

# Repack wheel
cd "$EXTRACT_DIR"
WHEEL_NAME=$(basename "$WHEEL")
zip -qr "$TEMP_DIR/$WHEEL_NAME" .

# Copy to destination
mkdir -p "$DEST_DIR"
cp "$TEMP_DIR/$WHEEL_NAME" "$DEST_DIR/"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Wheel repaired successfully: $DEST_DIR/$WHEEL_NAME"
