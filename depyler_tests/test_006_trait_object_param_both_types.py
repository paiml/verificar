import sys

def write_report(output, data: str):
    output.write(f"Report: {data}\n")
    output.write("=" * 40 + "\n")

def main():
    write_report(sys.stdout, "summary")
    with open("report.txt", "w") as f:
        write_report(f, "detailed")
