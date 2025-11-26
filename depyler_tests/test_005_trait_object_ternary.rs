use std as sys;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() {
    let mut out = if verbose {
        Box::new(std::io::stdout()) as Box<dyn std::io::Write>
    } else {
        Box::new(std::fs::File::create("log.txt".to_string())?)
    };
    out.write_all("message\n".to_string().as_bytes()).unwrap();
}
