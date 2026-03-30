# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys
import time

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

BASE_URL = "https://ftuxhxqka0.execute-api.us-east-2.amazonaws.com/api-gw-data-db-main/api/v1/data_db_main/"

max_attempts = int(os.environ.get("API_MAX_ATTEMPTS", "1"))
retry_interval = int(os.environ.get("API_RETRY_INTERVAL", "0"))
output_file = os.environ.get("API_OUTPUT_FILE", "")


def call_api():
    params = json.loads(os.environ["API_QUERY_PARAMS"])
    url = BASE_URL + os.environ["API_QUERY"]
    req = AWSRequest(method="GET", url=url, params=params)
    SigV4Auth(boto3.Session().get_credentials(), "execute-api", "us-east-2").add_auth(
        req
    )
    resp = requests.get(url, params=params, headers=dict(req.headers))
    resp.raise_for_status()
    return resp.json()


data = []
for attempt in range(1, max_attempts + 1):
    if max_attempts > 1:
        print(f"Attempt {attempt}/{max_attempts}", file=sys.stderr, flush=True)
    data = call_api()
    if data or max_attempts == 1:
        break
    if attempt < max_attempts:
        print(f"No data yet, waiting {retry_interval}s...", file=sys.stderr, flush=True)
        time.sleep(retry_interval)
else:
    print("Reached max attempts, proceeding with available data.", file=sys.stderr)

if output_file:
    with open(output_file, "w") as f:
        json.dump(data, f)
else:
    print(json.dumps(data))
