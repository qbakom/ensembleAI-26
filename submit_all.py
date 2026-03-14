"""
Submit all tasks to the hackathon server.
Must be run from Athena where the server is reachable.
"""
import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("TEAM_TOKEN")
SERVER_URL = os.getenv("SERVER_URL")

if not API_TOKEN or not SERVER_URL:
    print("ERROR: Set TEAM_TOKEN and SERVER_URL in .env")
    sys.exit(1)

headers = {"X-API-Token": API_TOKEN}


def submit_task(endpoint, file_path, file_key, description, form_data=None):
    print(f"\n{'='*60}")
    print(f"Submitting {description}")
    print(f"  Endpoint: {SERVER_URL}/{endpoint}")
    print(f"  File: {file_path}")

    if not os.path.exists(file_path):
        print(f"  SKIP: File not found!")
        return None

    print(f"  Size: {os.path.getsize(file_path)} bytes")

    try:
        response = requests.post(
            f"{SERVER_URL}/{endpoint}",
            files={file_key: open(file_path, "rb")},
            data=form_data,
            headers=headers,
            timeout=120
        )
        try:
            data = response.json()
        except Exception:
            data = response.text
        print(f"  Response: {response.status_code} {data}")
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def check_status(request_id):
    """Check submission status."""
    try:
        response = requests.get(
            f"{SERVER_URL}/status/{request_id}",
            headers=headers,
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3", "4"]
    results = {}

    if "1" in tasks:
        results["task1"] = submit_task(
            "task1",
            "task1/chebi_submission.parquet",
            "parquet_file",
            "Task 1: ChEBI Ontology Classification"
        )

    if "2" in tasks:
        # Submit both practice and public
        # Prefer smart predictions over multisignal
        for stage in ["practice", "public"]:
            base = "task2/EnsembleAI2026-starter-kit/predictions"
            for name in [f"python-{stage}-smart.jsonl", f"python-{stage}-multisignal.jsonl"]:
                path = os.path.join(base, name)
                if os.path.exists(path):
                    results[f"task2_{stage}"] = submit_task(
                        "task2", path, "jsonl_file",
                        f"Task 2: Code Context ({stage} - {os.path.basename(path)})",
                        form_data={"stage": stage}
                    )
                    break

    if "3" in tasks:
        # Try v2 first, fall back to v1
        v2_path = "task3/data/out/load_submission_v2.csv"
        v1_path = "task3/data/out/load_submission.csv"
        path = v2_path if os.path.exists(v2_path) else v1_path
        results["task3"] = submit_task(
            "task3",
            path,
            "csv_file",
            f"Task 3: Heat Pump Load Forecasting ({os.path.basename(path)})"
        )

    if "4" in tasks:
        results["task4"] = submit_task(
            "task4",
            "task4/data/out/ecg_submission.npz",
            "npz_file",
            "Task 4: ECG Digitization"
        )

    # Collect request IDs and poll status
    print(f"\n{'='*60}")
    print("Submission Results:")
    request_ids = {}
    for task, result in results.items():
        if result and isinstance(result, dict) and 'request_id' in result:
            request_ids[task] = result['request_id']
            print(f"  {task}: request_id={result['request_id']}")
        else:
            print(f"  {task}: {result}")

    if request_ids:
        print(f"\nWaiting 30s then checking status...")
        time.sleep(30)
        for task, rid in request_ids.items():
            status = check_status(rid)
            print(f"  {task} [{rid}]: {status}")


if __name__ == "__main__":
    main()
