import requests
import json

def check_api():
    base_url = "http://localhost:8000"
    results = {}
    
    try:
        r = requests.get(f"{base_url}/api/status", timeout=5)
        results["status"] = r.json()
    except Exception as e:
        results["status_error"] = str(e)

    try:
        r = requests.get(f"{base_url}/api/balance", timeout=5)
        results["balance"] = r.json()
    except Exception as e:
        results["balance_error"] = str(e)
        
    try:
        r = requests.get(f"{base_url}/api/logs", params={"limit": 100}, timeout=5)
        if r.status_code != 200:
             results["logs_error_body"] = r.text
             results["logs_error_code"] = r.status_code
        else:
             results["logs"] = r.json()
    except Exception as e:
        results["logs_error"] = str(e)

    with open("api_status.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    check_api()
