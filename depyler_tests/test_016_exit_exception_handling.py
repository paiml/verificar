class TransactionManager:
    def __init__(self):
        self.committed = False

    def __enter__(self):
        print("Starting transaction")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.committed = True
            print("Committed")
        else:
            print("Rolled back")
        return False

    def execute(self, query: str):
        print(f"Executing: {query}")

def main():
    with TransactionManager() as tx:
        tx.execute("INSERT INTO users VALUES (1)")
