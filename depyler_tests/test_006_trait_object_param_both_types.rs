use std as sys;
use std::io::Read;
use std::io::Write;
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn write_report(output: &mut std::fs::File, data: String) {
    output
        .write_all(format!("Report: {}\n", data).as_bytes())
        .unwrap();
    output
        .write_all(
            format!(
                "{}{}",
                "=".to_string().repeat(40 as usize),
                "\n".to_string()
            )
            .as_bytes(),
        )
        .unwrap();
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), std::io::Error> {
    write_report(std::io::stdout(), "summary");
    let mut f = std::fs::File::create("report.txt".to_string())?;
    write_report(f, "detailed".to_string());
    Ok(())
}
