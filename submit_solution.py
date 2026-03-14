"""Universal submission script for all tasks."""
import sys, os, requests

TEAM_TOKEN = os.environ.get("TEAM_TOKEN", "b2c6083ba78b4039a6db64a4bb5e07ca")
SERVER_URL = os.environ.get("SERVER_URL", "http://149.156.182.9:6060")

def submit(task, filepath, file_key, stage=None):
    headers = {"X-API-Token": TEAM_TOKEN}
    url = f"{SERVER_URL}/{task}"
    
    data = {}
    if stage:
        data["stage"] = stage
    
    print(f"Submitting {filepath} to {url} (key={file_key}, stage={stage})")
    
    with open(filepath, "rb") as f:
        resp = requests.post(url, files={file_key: f}, data=data if data else None, headers=headers)
    
    try:
        result = resp.json()
    except Exception:
        result = resp.text
    
    print(f"Status: {resp.status_code}")
    print(f"Response: {result}")
    return resp.status_code

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 submit_solution.py <task> <filepath> <file_key> [stage]")
        sys.exit(1)
    
    task = sys.argv[1]
    filepath = sys.argv[2]
    file_key = sys.argv[3]
    stage = sys.argv[4] if len(sys.argv) > 4 else None
    
    submit(task, filepath, file_key, stage)
