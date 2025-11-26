use std as sys;
#[derive(Debug, Clone)]
pub struct Logger {}
impl Logger {
    pub fn new(filename: String) -> Self {
        Self {}
    }
    pub fn log(&self, msg: String) {
        self.output.write(format!("[LOG] {}\n", msg));
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() {
    let logger = Logger::new();
    logger.log("test message".to_string());
}
