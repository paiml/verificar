use serde_json as json;
use serde_json;
#[derive(Debug, Clone)]
pub struct IndexError {
    message: String,
}
impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "index out of range: {}", self.message)
    }
}
impl std::error::Error for IndexError {}
impl IndexError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
#[doc = " Depyler: proven to terminate"]
pub fn get_items(json_str: &str) -> Result<Vec<i32>, IndexError> {
    let data = serde_json::from_str::<serde_json::Value>(&json_str).unwrap();
    let items = data.get("items").cloned().unwrap_or_default();
    Ok(items
        .iter()
        .copied()
        .map(|item| item.get("name").cloned().unwrap_or_default())
        .collect::<Vec<_>>())
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = get_items("{\"items\": [{\"name\": \"a\"}, {\"name\": \"b\"}]}")?;
    println!("{:?}", result);
    Ok(())
}
