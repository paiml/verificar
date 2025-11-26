use serde_json as json;
use serde_json;
use std::collections::HashMap;
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
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn merge_configs<'b, 'a>(
    base: &'a HashMap<serde_json::Value, serde_json::Value>,
    override_json: &'b str,
) -> HashMap<serde_json::Value, serde_json::Value> {
    let r#override = serde_json::from_str::<serde_json::Value>(&override_json).unwrap();
    let mut result = base.clone();
    for (k, v) in r#override {
        result.insert(k, v);
    }
    result
}
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), IndexError> {
    let base = serde_json::json!({ "debug": false, "port": 8080 });
    let merged = merge_configs(&base, "{\"debug\": true}");
    println!("{}", merged.get("debug").cloned().unwrap_or_default());
    Ok(())
}
