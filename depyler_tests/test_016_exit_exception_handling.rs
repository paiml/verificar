use serde_json;
#[derive(Debug, Clone)]
pub struct TransactionManager {
    pub committed: bool,
}
impl TransactionManager {
    pub fn new() -> Self {
        Self { committed: false }
    }
    pub fn __enter__(&self) -> &Self {
        println!("{}", "Starting transaction".to_string());
        return self;
    }
    pub fn __exit__(
        &mut self,
        exc_type: serde_json::Value,
        exc_val: serde_json::Value,
        exc_tb: serde_json::Value,
    ) -> bool {
        if exc_type.is_none() {
            self.committed = true;
            println!("{}", "Committed".to_string());
        } else {
            println!("{}", "Rolled back".to_string());
        };
        return false;
    }
    pub fn execute(&self, query: String) {
        println!("{}", format!("Executing: {}", query));
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() {
    let _context = TransactionManager::new();
    let tx = _context.__enter__();
    tx.execute("INSERT INTO users VALUES(1)".to_string());
}
