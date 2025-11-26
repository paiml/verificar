import json

def parse_config(json_str: str) -> str:
    config = json.loads(json_str)
    name = config["name"]
    return name

def main():
    result = parse_config('{"name": "test"}')
    print(result)
