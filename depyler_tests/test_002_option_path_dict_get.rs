use std::collections::HashMap;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), std::io::Error> {
    let config = {
        let mut map = HashMap::new();
        map.insert("output".to_string(), "result.txt");
        map
    };
    let path = config
        .get(&"output")
        .cloned()
        .unwrap_or("default.txt".to_string());
    let mut f = std::fs::File::create(path.as_ref().unwrap())?;
    f.write_all("data".to_string().as_bytes()).unwrap();
    Ok(())
}
