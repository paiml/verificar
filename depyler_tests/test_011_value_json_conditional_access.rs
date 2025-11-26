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
pub fn process_response(json_str: &str) -> Result<i32, IndexError> {
    let response = serde_json::from_str::<serde_json::Value>(&json_str).unwrap();
    let _cse_temp_0 = response.get("status").cloned().unwrap_or_default() == "ok";
    if _cse_temp_0 {
        let data = response.get("data").cloned().unwrap_or_default();
        let count = data.get("count").cloned().unwrap_or_default();
        return Ok(count);
    }
    Ok(0)
}
#[doc = " Depyler: verified panic-free"]
#[doc = " Depyler: proven to terminate"]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = process_response("{\"status\": \"ok\", \"data\": {\"count\": 42}}")?;
    println!("{}", result);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{quickcheck, TestResult};
    #[test]
    fn test_process_response_examples() {
        assert_eq!(process_response(""), 0);
        assert_eq!(process_response("a"), 1);
        assert_eq!(process_response("abc"), 3);
    }
}
