import json

def merge_configs(base: dict, override_json: str) -> dict:
    override = json.loads(override_json)
    result = base.copy()
    result.update(override)
    return result

def main():
    base = {"debug": False, "port": 8080}
    merged = merge_configs(base, '{"debug": true}')
    print(merged["debug"])
