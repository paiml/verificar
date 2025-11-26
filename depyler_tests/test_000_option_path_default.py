def process_file(filename: str = None):
    if filename is None:
        filename = "default.txt"
    with open(filename, "r") as f:
        return f.read()

def main():
    result = process_file()
    print(result)
