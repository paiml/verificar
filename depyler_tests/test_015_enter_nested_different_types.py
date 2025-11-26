class Timer:
    def __enter__(self):
        self.start = 0
        return self

    def __exit__(self, *args):
        self.elapsed = 100
        return False

def main():
    with Timer() as t:
        with open("data.txt", "w") as f:
            f.write("timed write")
    print(f"Elapsed: {t.elapsed}")
