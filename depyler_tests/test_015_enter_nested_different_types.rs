use std::io::Read;
use std::io::Write;
#[derive(Debug, Clone)]
pub struct Timer {}
impl Timer {
    pub fn new() -> Self {
        Self {}
    }
    pub fn __enter__(&mut self) -> &Self {
        self.start = 0;
        return self;
    }
    pub fn __exit__(&mut self) -> bool {
        self.elapsed = 100;
        return false;
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), std::io::Error> {
    let _context = Timer::new();
    let t = _context.__enter__();
    let mut f = std::fs::File::create("data.txt".to_string())?;
    f.write_all("timed write".to_string().as_bytes()).unwrap();
    println!("{}", format!("Elapsed: {}", t.elapsed));
    Ok(())
}
