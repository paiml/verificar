class Connection:
    def __init__(self, host: str):
        self.host = host
        self.connected = False

    def __enter__(self):
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
        return False

    def query(self, sql: str) -> str:
        return f"Result from {self.host}"

def main():
    with Connection("localhost") as conn:
        result = conn.query("SELECT 1")
        print(result)
