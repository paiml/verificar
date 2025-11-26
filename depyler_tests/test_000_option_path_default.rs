use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn process_file(mut filename: Option<String>) -> Result<(), std::io::Error> {
    if filename.is_none() {
        filename = "default.txt";
    }
    let mut f = std::fs::File::open(filename.as_ref().unwrap())?;
    return Ok({
        let mut content = String::new();
        f.read_to_string(&mut content)?;
        content
    });
    Ok(())
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = process_file()?;
    println!("{}", result);
    Ok(())
}
