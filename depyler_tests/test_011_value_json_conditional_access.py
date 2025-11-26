import json

def process_response(json_str: str):
    response = json.loads(json_str)
    if response["status"] == "ok":
        data = response["data"]
        count = data["count"]
        return count
    return 0

def main():
    result = process_response('{"status": "ok", "data": {"count": 42}}')
    print(result)
