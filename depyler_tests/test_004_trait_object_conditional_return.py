import sys

def get_writer(use_stdout: bool):
    if use_stdout:
        return sys.stdout
    return open("output.txt", "w")

def main():
    writer = get_writer(True)
    writer.write("hello\n")
