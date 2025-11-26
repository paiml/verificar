def main():
    config = {"output": "result.txt"}
    path = config.get("output", "default.txt")
    with open(path, "w") as f:
        f.write("data")
