use serde_json;
#[derive(Debug, Clone)]
pub struct Connection {
    pub host: String,
    pub connected: bool,
}
impl Connection {
    pub fn new(host: String) -> Self {
        Self {
            host,
            connected: false,
        }
    }
    pub fn __enter__(&mut self) -> &Self {
        self.connected = true;
        return self;
    }
    pub fn __exit__(
        &mut self,
        exc_type: serde_json::Value,
        exc_val: serde_json::Value,
        exc_tb: serde_json::Value,
    ) -> bool {
        self.connected = false;
        return false;
    }
    pub fn query(&self, sql: String) -> String {
        return format!("Result from {}", self.host);
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() {
    let _context = Connection::new("localhost".to_string().to_string());
    let conn = _context.__enter__();
    let result = conn.query("SELECT 1".to_string());
    println!("{}", result);
}
