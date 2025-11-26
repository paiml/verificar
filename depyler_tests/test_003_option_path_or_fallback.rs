use std as os;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), std::io::Error> {
    let _cse_temp_0 = (std::env::var("OUTPUT_FILE".to_string()).ok()) || ("output.txt");
    let path = _cse_temp_0;
    let mut f = std::fs::File::create(&path)?;
    f.write_all("result".to_string().as_bytes()).unwrap();
    Ok(())
}
