import sys

class Logger:
    def __init__(self, filename: str = None):
        if filename:
            self.output = open(filename, "w")
        else:
            self.output = sys.stdout

    def log(self, msg: str):
        self.output.write(f"[LOG] {msg}\n")

def main():
    logger = Logger()
    logger.log("test message")
