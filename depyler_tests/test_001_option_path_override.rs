use std::io::Read;
    use std::io::Write;
    #[doc = " Depyler: verified panic-free"] #[doc = " Depyler: proven to terminate"] pub fn get_config_path(override: & Option<String>) -> String {
    if r#override.is_some() {
    return r#override.to_string();
   
}
"config.json".to_string()
}
#[doc = " Depyler: verified panic-free"] #[doc = " Depyler: proven to terminate"] pub fn main () -> Result <(), std::io::Error>{
    let path = get_config_path();
    let mut f = std::fs::File::open(& path) ?;
    println!("{}", {
    let mut content = String::new();
    f.read_to_string(&mut content) ?;
    content });
    Ok(()) }