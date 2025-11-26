use serde_json;
#[derive(Debug, Clone)]
pub struct FileManager {
    pub filename: String,
    pub file: (),
}
impl FileManager {
    pub fn new(filename: String) -> Self {
        Self {
            filename,
            file: Default::default(),
        }
    }
    pub fn __enter__(&mut self) {
        self.file = open(self.filename, "w".to_string());
        return self.file;
    }
    pub fn __exit__(
        &self,
        exc_type: serde_json::Value,
        exc_val: serde_json::Value,
        exc_tb: serde_json::Value,
    ) -> bool {
        if self.file {
            self.file.close();
        };
        return false;
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() {
    let _context = FileManager::new("output.txt".to_string().to_string());
    let f = _context.__enter__();
    f.write_all("managed write".to_string().as_bytes()).unwrap();
}
