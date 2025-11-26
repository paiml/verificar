import os

def main():
    path = os.environ.get("OUTPUT_FILE") or "output.txt"
    with open(path, "w") as f:
        f.write("result")
