import sys

def broadcast(writers: list, msg: str):
    for w in writers:
        w.write(msg)

def main():
    with open("log.txt", "w") as f:
        broadcast([sys.stdout, f], "broadcast message\n")
