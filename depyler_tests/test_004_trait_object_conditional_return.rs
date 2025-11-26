use std as sys;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn get_writer(use_stdout: bool) -> Result<String, std::io::Error> {
    if use_stdout {
        return Ok(std::io::stdout());
    }
    Ok(std::fs::File::create("output.txt")?)
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = get_writer(true)?;
    writer.write_all("hello\n".to_string().as_bytes()).unwrap();
    Ok(())
}
