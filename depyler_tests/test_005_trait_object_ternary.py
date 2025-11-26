import sys

def main():
    verbose = True
    out = sys.stdout if verbose else open("log.txt", "w")
    out.write("message\n")
