# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Command-line tool to install dependencies for the tt-forge package.
"""

import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def main():
    """
    Install dependencies for the tt-forge package.
    """

    def find_name_value(content, name):
        match = re.search(rf"^{re.escape(name)}\s*=\s*(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    try:
        # Find the package installation directory
        import pjrt_plugin_tt

        package_dir = Path(pjrt_plugin_tt.__file__).parent

        # Construct path to sfpi-version file
        sfpi_version_path = package_dir / "tt-metal" / "tt_metal" / "sfpi-version"

        if not sfpi_version_path.exists():
            print(
                f"Error: sfpi-version file not found at {sfpi_version_path}",
                file=sys.stderr,
            )
            return 1

        # Read and print the contents
        with open(sfpi_version_path, "r") as f:
            content = f.read().strip()

        print("SFPI Version Information:")
        print("-" * 40)
        print(content)
        sfpi_repo = find_name_value(content, "sfpi_repo")
        sfpi_version = find_name_value(content, "sfpi_version")
        sfpi_arch = "x86_64"
        # Detect Linux distribution type
        sfpi_dist = "debian"
        sfpi_pkg = "deb"
        try:
            with open("/etc/os-release", "r") as f:
                os_release = f.read().lower()
                if (
                    "fedora" in os_release
                    or "rhel" in os_release
                    or "centos" in os_release
                    or "suse" in os_release
                    or "mandriva" in os_release
                ):
                    print("Detected rpm linux distribution")
                    sfpi_dist = "fedora"
                    sfpi_pkg = "rpm"
                elif "debian" in os_release or "ubuntu" in os_release:
                    print("Detected debian linux distribution")
                else:
                    print(
                        "Warning: Unknown distribution, defaulting to debian",
                        file=sys.stderr,
                    )
        except FileNotFoundError:
            print(
                "Warning: Could not detect distribution, defaulting to debian",
                file=sys.stderr,
            )

        sfpi_url = f"{sfpi_repo}/releases/download/{sfpi_version}/sfpi_{sfpi_version}_{sfpi_arch}_{sfpi_dist}.{sfpi_pkg}"

        # Download the package
        print(f"Downloading SFPI package from {sfpi_url}...")
        temp_dir = tempfile.mkdtemp()
        temp_pkg_path = (
            Path(temp_dir) / f"sfpi_{sfpi_version}_{sfpi_arch}_{sfpi_dist}.{sfpi_pkg}"
        )

        try:
            urllib.request.urlretrieve(sfpi_url, temp_pkg_path)
            print(f"Downloaded to {temp_pkg_path}")

            # Install the package using dpkg
            print("Installing SFPI package...")
            if sfpi_dist == "debian":
                subprocess.run(["sudo", "dpkg", "-i", str(temp_pkg_path)], check=True)
            elif sfpi_dist == "fedora":
                subprocess.run(["sudo", "rpm", "-i", str(temp_pkg_path)], check=True)
            print("SFPI package installed successfully!")

        except urllib.error.URLError as e:
            print(f"Error downloading package: {e}", file=sys.stderr)
            return 1
        except subprocess.CalledProcessError as e:
            print(f"Error installing package: {e}", file=sys.stderr)
            print("You may need to run: sudo apt-get install -f", file=sys.stderr)
            return 1
        finally:
            # Clean up temporary file
            if temp_pkg_path.exists():
                temp_pkg_path.unlink()
            Path(temp_dir).rmdir()
        return 0

    except ImportError:
        print(
            "Error: pjrt_plugin_tt package not found. Please install the package first.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
