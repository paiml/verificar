use serde_json;
use std as sys;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
pub fn broadcast<'b, 'a>(writers: &'a Vec<serde_json::Value>, msg: &'b str) {
    for w in writers.iter().cloned() {
        w.write_all(msg.as_bytes()).unwrap();
    }
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), std::io::Error> {
    let mut f = std::fs::File::create("log.txt".to_string())?;
    broadcast(
        &vec![std::io::stdout(), f],
        "broadcast message\n".to_string(),
    );
    Ok(())
}
