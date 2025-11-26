import json

def get_items(json_str: str) -> list:
    data = json.loads(json_str)
    items = data["items"]
    return [item["name"] for item in items]

def main():
    result = get_items('{"items": [{"name": "a"}, {"name": "b"}]}')
    print(result)
