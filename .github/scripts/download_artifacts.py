# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt

import requests


def get_token(cli_token):
    # Priority 1: Command-line argument
    if cli_token:
        return cli_token

    # Priority 2: Check for a token file in the home directory (~/.ghtoken)
    token_file = os.path.expanduser("~/.ghtoken")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r") as f:
                token = f.read().strip()
                if token:
                    return token
        except Exception as e:
            print("Error reading token from ~/.ghtoken:", e)

    # Priority 3: Environment variable
    env_token = os.environ.get("GITHUB_TOKEN")
    if env_token:
        return env_token

    # Priority 3.5: Environment variable (as commonly used in our CI)
    env_token = os.environ.get("GH_TOKEN")
    if env_token:
        return env_token

    # Priority 4: Fallback to GitHub CLI (gh)
    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if token:
            return token
    except Exception as e:
        print("Could not retrieve token using GitHub CLI:", e)

    return None


def download_artifact(artifact, folder_name, headers, args, session):
    """
    Download the artifact ZIP file and return its file path.
    Unzipping is deferred.
    """
    artifact_name = artifact["name"]
    if artifact_name in ["install-artifacts"]:
        return None
    if args.filter and args.filter not in artifact_name:
        return None

    artifact_id = artifact["id"]
    # Destination ZIP file: use artifact_name + ".zip"
    target_zip = os.path.join(folder_name, f"{artifact_name}.zip")
    print(f"Downloading artifact '{artifact_name}' (ID: {artifact_id}) to {target_zip}")
    download_url = (
        f"https://api.github.com/repos/{args.repo}/actions/artifacts/{artifact_id}/zip"
    )
    try:
        with session.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(target_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return target_zip
    except Exception as e:
        print(f"Failed to download artifact '{artifact_name}': {e}")
        return None


def recursive_unzip(directory):
    """
    Recursively search for and extract any ZIP files in the given directory.
    After extracting a ZIP file, it is removed.
    """
    zip_found = True
    while zip_found:
        zip_found = False
        # Walk the directory to find any .zip file.
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"Recursively unzipping {zip_path}")
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(root)
                        os.remove(zip_path)
                        zip_found = True
                    except Exception as e:
                        print(f"Failed to recursively unzip {zip_path}: {e}")


def process_zip_file(zip_path, folder_name):
    """
    Unzip the given ZIP file. For ZIP files with names like
    "full-logs-<model_name>.zip", if the extracted contents include
    "full_job_output.log", rename it to "<model_name>_full_job_output.log".
    Then, recursively extract any nested ZIP files.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = [zi for zi in zip_ref.infolist() if not zi.is_dir()]
            zip_ref.extractall(folder_name)
        os.remove(zip_path)

        zip_base = os.path.basename(zip_path)
        # Check for the special pattern "full-logs-<model_name>.zip"
        if zip_base.startswith("full-logs-") and zip_base.endswith(".zip"):
            # Extract <model_name> from the zip filename.
            # E.g., "full-logs-vision-misc.zip" gives model_name = "vision-misc"
            model_name = zip_base[len("full-logs-") : -len(".zip")]
            extracted_log = os.path.join(folder_name, "full_job_output.log")
            if os.path.exists(extracted_log):
                new_name = os.path.join(
                    folder_name, f"{model_name}_full_job_output.log"
                )
                os.rename(extracted_log, new_name)
                print(f"Renamed {extracted_log} to {new_name}")

        # Temporary hack to get p150 and n150 in filename:
        if zip_base.startswith("test-reports-") and zip_base.endswith(".zip"):
            arch_name = zip_base.split("-")[4]

            # Special case for n300-llmbox which contains dash
            if "n300-llmbox" in zip_base:
                arch_name = "n300-llmbox"
            # print(f"Arch name: {arch_name}")

            # Full paths of extracted files
            extracted_names = [m.filename for m in members]

            # Do the rename for all files in zip to include arch name
            for extracted_name in extracted_names:
                new_name = f"{arch_name}_{extracted_name}"
                new_path = os.path.join(folder_name, new_name)
                extracted_path = os.path.join(folder_name, extracted_name)
                os.rename(extracted_path, new_path)
                print(f"Renamed {extracted_path} to {new_path}")

    except Exception as e:
        print(f"Failed to process zip file '{zip_path}': {e}")


def get_run_details(args, headers, workflow_name, run_index_n=0):
    """
    Retrieves run details either from a specified run-id or by fetching the latest run.
    Returns a tuple of (run_id, date_str, folder_name).

    run_index_n: If not specifying a run-id, specify the index of the nth latest run to download artifacts from.
    0 is the latest run, 1 is the second latest, etc.
    """
    if args.run_id:

        assert run_index_n == 0, "run_index_n is incompatible with run_id"

        run_id = args.run_id
        run_detail_url = (
            f"https://api.github.com/repos/{args.repo}/actions/runs/{run_id}"
        )
        run_response = requests.get(run_detail_url, headers=headers)
        if run_response.status_code != 200:
            print(f"Failed to get details for run {run_id}: {run_response.text}")
            exit(1)
        run_data = run_response.json()
        created_at = run_data["created_at"]
        date_str = dt.fromisoformat(created_at.rstrip("Z")).strftime("%Y%m%d")
    else:
        print("Fetching workflow runs for", workflow_name)
        runs_url = f"https://api.github.com/repos/{args.repo}/actions/workflows/{args.workflow}/runs"
        params = {
            "branch": args.branch,
            "per_page": run_index_n + 1,
        }  # fetch up to the specified latest run
        runs_response = requests.get(runs_url, headers=headers, params=params)
        if runs_response.status_code != 200:
            print("Failed to get workflow runs:", runs_response.text)
            exit(1)
        runs_data = runs_response.json()
        if not runs_data.get("workflow_runs"):
            print(f"No workflow runs found on branch '{args.branch}'")
            exit(1)
        try:
            nth_latest_run = runs_data["workflow_runs"][run_index_n]
        except IndexError:
            print(f"Error: No run found at index {run_index_n}.")
            print(
                "\tHelp: If not specifying a run-id, specify the index of the latest run to download artifacts from. 0 is the latest run, 1 is the second latest, etc."
            )
            exit(1)
        run_id = nth_latest_run["id"]
        created_at = nth_latest_run["created_at"]
        date_str = dt.fromisoformat(created_at.rstrip("Z")).strftime("%Y%m%d")
    folder_name = f"{workflow_name}_artifacts_{date_str}_run_id_{run_id}"
    return run_id, date_str, folder_name


def fetch_artifacts(repo, run_id, headers):
    """
    Fetches all artifacts for a given run_id using pagination.
    Returns a list of artifact dictionaries.
    """
    artifacts_url = (
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts"
    )
    all_artifacts = []
    page = 1
    while True:
        params = {"per_page": 100, "page": page}
        artifacts_response = requests.get(artifacts_url, headers=headers, params=params)
        if artifacts_response.status_code != 200:
            print("Failed to get artifacts:", artifacts_response.text)
            exit(1)
        artifacts_data = artifacts_response.json()
        artifacts = artifacts_data.get("artifacts", [])
        if not artifacts:
            break
        all_artifacts.extend(artifacts)
        if len(artifacts) < 100:
            break
        page += 1
    return all_artifacts


def deduplicate_artifacts(artifacts):
    """
    Deduplicate artifacts by name, keeping only the one with the latest updated_at timestamp.
    If updated_at is missing, created_at is used.
    """
    artifact_map = {}
    for artifact in artifacts:
        name = artifact["name"]
        # Use updated_at if available, otherwise created_at.
        timestamp_str = artifact.get("updated_at", artifact.get("created_at"))
        try:
            timestamp = dt.fromisoformat(timestamp_str.rstrip("Z"))
        except Exception:
            timestamp = dt.min
        if name in artifact_map:
            existing_timestamp_str = artifact_map[name].get(
                "updated_at", artifact_map[name].get("created_at")
            )
            try:
                existing_timestamp = dt.fromisoformat(
                    existing_timestamp_str.rstrip("Z")
                )
            except Exception:
                existing_timestamp = dt.min
            if timestamp > existing_timestamp:
                artifact_map[name] = artifact
        else:
            artifact_map[name] = artifact
    return list(artifact_map.values())


def list_artifacts(artifacts, args):
    """
    Lists artifacts based on the provided filter and arguments.
    """
    for artifact in artifacts:
        artifact_name = artifact["name"]
        if artifact_name in ["install-artifacts"]:
            continue
        if args.filter and args.filter not in artifact_name:
            continue
        artifact_id = artifact["id"]
        download_url = f"https://api.github.com/repos/{args.repo}/actions/artifacts/{artifact_id}/zip"
        print(
            f"Found artifact '{artifact_name}' (ID: {artifact_id}) download_url: {download_url}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions artifacts for a specific run or the latest run on a given branch."
    )
    parser.add_argument(
        "--repo",
        default="tenstorrent/tt-xla",
        help="Repository in owner/repo format (default: tenstorrent/tt-xla)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional: GitHub run-id to download artifacts from instead of specifying branch and workflow",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to filter workflow runs (default: main)",
    )
    parser.add_argument(
        "--workflow",
        default="nightly-tests.yml",
        help="Optional: Specify the workflow file name to filter runs (default: nightly-tests.yml)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Optional: Filter artifacts by a substring in their name",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub Personal Access Token (if not provided, the script will try to retrieve one using GitHub CLI or environment variable)",
    )
    parser.add_argument(
        "--list", action="store_true", help="Just list artifacts without downloading"
    )
    parser.add_argument(
        "--no-unzip",
        dest="unzip",
        action="store_false",
        help="Do not unzip downloaded .zip files (default is to unzip, including nested ZIPs. Removes original ZIP and applies special renaming if needed.)",
    )
    parser.set_defaults(unzip=True)
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        help="Output directory for artifacts",
    )

    parser.add_argument(
        "--run-lookback-idx",
        type=int,
        default=0,
        help="If not specifying a run-id, specify the index of the latest run to download artifacts from. 0 is the latest run, 1 is the second latest, etc.",
    )

    args = parser.parse_args()

    token = get_token(args.token)
    if not token:
        print(
            "Error: GitHub token is required. Provide it with --token, set GITHUB_TOKEN, or login via 'gh auth login'."
        )
        exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Remove the .yml/.yaml extension from the workflow filename for folder naming.
    workflow_name = os.path.splitext(args.workflow)[0]

    # Retrieve run details (either specific run-id or the latest run)
    run_id, date_str, folder_name = get_run_details(
        args, headers, workflow_name, run_index_n=args.run_lookback_idx
    )

    folder_name = args.output_folder if args.output_folder else folder_name

    if args.list:
        print(f"Listing artifacts for run {run_id}.")
    else:
        os.makedirs(folder_name, exist_ok=True)
        print(
            f"Downloading artifacts for run {run_id} (created on {date_str}) into folder '{folder_name}'."
        )

    # Fetch artifacts for the specified run.
    all_artifacts = fetch_artifacts(args.repo, run_id, headers)
    if not all_artifacts:
        print("No artifacts found for run", run_id)
        exit(0)

    # Deduplicate artifacts by name to avoid duplicate downloads.
    all_artifacts = deduplicate_artifacts(all_artifacts)

    # If --list is set, list artifact information and exit.
    if args.list:
        list_artifacts(all_artifacts, args)
        exit(0)

    downloaded_zips = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = [
                executor.submit(
                    download_artifact, artifact, folder_name, headers, args, session
                )
                for artifact in all_artifacts
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:  # result is the file path for the downloaded ZIP
                    downloaded_zips.append(result)

    # If --unzip is set, process each ZIP file one-at-a-time.
    if args.unzip:
        for zip_file in downloaded_zips:
            process_zip_file(zip_file, folder_name)
        # If any zip files remain (recursive zips) extract them now.
        recursive_unzip(folder_name)


if __name__ == "__main__":
    main()
