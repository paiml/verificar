def get_config_path(override: str = None) -> str:
    if override:
        return override
    return "config.json"

def main():
    path = get_config_path()
    with open(path, "r") as f:
        print(f.read())
