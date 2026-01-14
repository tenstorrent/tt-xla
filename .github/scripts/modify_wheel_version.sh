#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

python3 -m venv venv
source venv/bin/activate
pip install wheel

wheel_path="$(realpath $1)"
wheel_dir="$(dirname $1)"
wheel_version="$2"

pushd $wheel_dir

# Exploded all wheel packages
wheel unpack "$wheel_path"
# Delete old wheel
rm "$wheel_path"

# Update the METADATA file with new version.
perl  -X -p -i -e  's!(?<=^Version:\s)[\w\.\+]+!'$wheel_version'!g' */*/METADATA

root_folders=$(find . -maxdepth 1 -type d -printf '%f\n' | grep -P '(?<=\-)[\w\.\+]+' | xargs )

# Rename folders root folder to match new version
for rf in $root_folders; do
    package_prefix="$(echo $rf| grep -oP '^\w+-')"
    new_folder_name="$(echo $rf| perl -X -p -e 's!(?<=\-)[\w\.\+]+!'$wheel_version'!g')"
    mv $rf $new_folder_name
    pushd $new_folder_name
    ls
    child_folders=$(find . -maxdepth 1 -type d -printf '%f\n' | grep -P '(?<=\-)[\w\.\+]+(?=\.dist-info|\.data)' | xargs )
    # Rename folders with .dist-info and/or .data to match new version
    for cf in $child_folders; do
        new_folder_name="$(echo $cf| perl -X -p -e 's!(?<=\-)[\w\.\+]+(?=\.dist-info|\.data)!'$wheel_version'!g')"
        mv $cf $new_folder_name
    done
    popd
    ls
    # Repack wheel
    ls | grep "$package_prefix$wheel_version" | xargs -I{} wheel pack {}
    # Delete old exploded wheel folder
    ls -d $package_prefix*$wheel_version | xargs -I{} rm -r {}
    ls
done

popd
