//! Entrenar LLM fine-tuning integration
//!
//! Exports verified transpilation tuples for LoRA fine-tuning with entrenar.
//! See VERIFICAR-090.

use crate::data::VerifiedTuple;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training example for code-to-code translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTranslationExample {
    /// Unique identifier
    pub id: String,
    /// Source language
    pub source_language: String,
    /// Target language
    pub target_language: String,
    /// Source code
    pub source_code: String,
    /// Target code (correct translation)
    pub target_code: String,
    /// Prompt for LLM (formatted input)
    pub prompt: String,
    /// Completion for LLM (expected output)
    pub completion: String,
}

/// Prompt template for code translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template name
    pub name: String,
    /// System prompt (optional)
    pub system: Option<String>,
    /// User prompt template with placeholders: {source_lang}, {target_lang}, {source_code}
    pub user_template: String,
    /// Whether to include language tags
    pub include_lang_tags: bool,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self::instruction_following()
    }
}

impl PromptTemplate {
    /// Instruction-following style prompt (Alpaca/Vicuna format)
    #[must_use]
    pub fn instruction_following() -> Self {
        Self {
            name: "instruction".to_string(),
            system: Some("You are an expert code translator.".to_string()),
            user_template: "Translate the following {source_lang} code to {target_lang}:\n\n```{source_lang}\n{source_code}\n```".to_string(),
            include_lang_tags: true,
        }
    }

    /// Chat-style prompt (ChatML format)
    #[must_use]
    pub fn chat_style() -> Self {
        Self {
            name: "chat".to_string(),
            system: Some("You are a helpful assistant that translates code between programming languages.".to_string()),
            user_template: "Please convert this {source_lang} code to idiomatic {target_lang}:\n\n{source_code}".to_string(),
            include_lang_tags: false,
        }
    }

    /// Completion-style prompt (minimal, for base models)
    #[must_use]
    pub fn completion_style() -> Self {
        Self {
            name: "completion".to_string(),
            system: None,
            user_template: "# {source_lang}\n{source_code}\n\n# {target_lang}\n".to_string(),
            include_lang_tags: false,
        }
    }

    /// Format a prompt using this template
    #[must_use]
    pub fn format(&self, source_lang: &str, target_lang: &str, source_code: &str) -> String {
        self.user_template
            .replace("{source_lang}", source_lang)
            .replace("{target_lang}", target_lang)
            .replace("{source_code}", source_code)
    }

    /// Format completion (target code with optional language tag)
    #[must_use]
    pub fn format_completion(&self, target_lang: &str, target_code: &str) -> String {
        if self.include_lang_tags {
            format!("```{target_lang}\n{target_code}\n```")
        } else {
            target_code.to_string()
        }
    }
}

/// Export configuration for entrenar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output format: "json", "jsonl", "parquet"
    pub format: ExportFormat,
    /// Prompt template to use
    pub template: PromptTemplate,
    /// Train/val split ratio (0.0 to 1.0)
    pub train_ratio: f64,
    /// Random seed for splitting
    pub seed: u64,
    /// Maximum examples to export (None for all)
    pub max_examples: Option<usize>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Jsonl,
            template: PromptTemplate::default(),
            train_ratio: 0.9,
            seed: 42,
            max_examples: None,
        }
    }
}

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// Single JSON array
    Json,
    /// JSON Lines (one object per line)
    Jsonl,
    /// Apache Parquet
    Parquet,
}

/// Export statistics
#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    /// Total examples exported
    pub total: usize,
    /// Training examples
    pub train_count: usize,
    /// Validation examples
    pub val_count: usize,
    /// Average source code length
    pub avg_source_len: f64,
    /// Average target code length
    pub avg_target_len: f64,
}

/// Exporter for entrenar training data
#[derive(Debug)]
pub struct EntrenarExporter {
    config: ExportConfig,
}

impl EntrenarExporter {
    /// Create a new exporter with configuration
    #[must_use]
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Convert verified tuple to training example
    #[must_use]
    pub fn to_example(&self, tuple: &VerifiedTuple, id: &str) -> CodeTranslationExample {
        let source_lang = tuple.source_language.to_string();
        let target_lang = tuple.target_language.to_string();

        let prompt = self
            .config
            .template
            .format(&source_lang, &target_lang, &tuple.source_code);
        let completion = self
            .config
            .template
            .format_completion(&target_lang, &tuple.target_code);

        CodeTranslationExample {
            id: id.to_string(),
            source_language: source_lang,
            target_language: target_lang,
            source_code: tuple.source_code.clone(),
            target_code: tuple.target_code.clone(),
            prompt,
            completion,
        }
    }

    /// Export verified tuples to training data
    ///
    /// # Errors
    ///
    /// Returns error if export fails
    pub fn export(
        &self,
        tuples: &[VerifiedTuple],
        output_dir: &Path,
    ) -> std::io::Result<ExportStats> {
        let examples: Vec<_> = tuples
            .iter()
            .take(self.config.max_examples.unwrap_or(usize::MAX))
            .enumerate()
            .map(|(i, t)| self.to_example(t, &format!("ex_{i:06}")))
            .collect();

        let (train, val) = self.split_train_val(&examples);

        let stats = ExportStats {
            total: examples.len(),
            train_count: train.len(),
            val_count: val.len(),
            avg_source_len: examples
                .iter()
                .map(|e| e.source_code.len())
                .sum::<usize>() as f64
                / examples.len().max(1) as f64,
            avg_target_len: examples
                .iter()
                .map(|e| e.target_code.len())
                .sum::<usize>() as f64
                / examples.len().max(1) as f64,
        };

        std::fs::create_dir_all(output_dir)?;

        match self.config.format {
            ExportFormat::Json => {
                self.write_json(&train, &output_dir.join("train.json"))?;
                self.write_json(&val, &output_dir.join("val.json"))?;
            }
            ExportFormat::Jsonl => {
                self.write_jsonl(&train, &output_dir.join("train.jsonl"))?;
                self.write_jsonl(&val, &output_dir.join("val.jsonl"))?;
            }
            ExportFormat::Parquet => {
                // Parquet export requires additional dependencies
                // For now, fall back to JSONL
                self.write_jsonl(&train, &output_dir.join("train.jsonl"))?;
                self.write_jsonl(&val, &output_dir.join("val.jsonl"))?;
            }
        }

        Ok(stats)
    }

    /// Split examples into train/val sets
    fn split_train_val(
        &self,
        examples: &[CodeTranslationExample],
    ) -> (Vec<CodeTranslationExample>, Vec<CodeTranslationExample>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut train = Vec::new();
        let mut val = Vec::new();

        for (i, example) in examples.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            (self.config.seed, i).hash(&mut hasher);
            let hash = hasher.finish();

            #[allow(clippy::cast_sign_loss)]
            let threshold = (self.config.train_ratio * u64::MAX as f64) as u64;

            if hash < threshold {
                train.push(example.clone());
            } else {
                val.push(example.clone());
            }
        }

        (train, val)
    }

    fn write_json(&self, examples: &[CodeTranslationExample], path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(examples)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    fn write_jsonl(&self, examples: &[CodeTranslationExample], path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        for example in examples {
            let line = serde_json::to_string(example)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }
}

/// Generate entrenar YAML config for the exported data
#[must_use]
pub fn generate_entrenar_config(
    data_dir: &Path,
    output_dir: &Path,
    lora_rank: usize,
) -> String {
    format!(
        r"# Entrenar configuration for verificar training data
# Generated by verificar v{}

model:
  path: codellama-7b.gguf  # Replace with your base model
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: {}
  val: {}
  batch_size: 4
  seq_len: 2048

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

lora:
  rank: {}
  alpha: {}
  target_modules: [q_proj, v_proj]
  dropout: 0.05

training:
  epochs: 3
  grad_clip: 1.0
  lr_scheduler: cosine
  warmup_steps: 100
  save_interval: 1
  output_dir: {}
",
        env!("CARGO_PKG_VERSION"),
        data_dir.join("train.jsonl").display(),
        data_dir.join("val.jsonl").display(),
        lora_rank,
        lora_rank * 2, // alpha = 2 * rank is common
        output_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Language;

    fn sample_tuple() -> VerifiedTuple {
        VerifiedTuple {
            source_language: Language::Python,
            target_language: Language::Rust,
            source_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
            target_code: "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string(),
            is_correct: true,
            execution_time_ms: 10,
        }
    }

    #[test]
    fn test_prompt_template_instruction() {
        let template = PromptTemplate::instruction_following();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("Python"));
        assert!(prompt.contains("Rust"));
        assert!(prompt.contains("x = 1"));
        assert!(prompt.contains("```Python"));
    }

    #[test]
    fn test_prompt_template_chat() {
        let template = PromptTemplate::chat_style();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("Python"));
        assert!(prompt.contains("idiomatic Rust"));
        assert!(!prompt.contains("```")); // No code blocks in chat style
    }

    #[test]
    fn test_prompt_template_completion() {
        let template = PromptTemplate::completion_style();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("# Python"));
        assert!(prompt.contains("# Rust"));
    }

    #[test]
    fn test_format_completion_with_tags() {
        let template = PromptTemplate::instruction_following();
        let completion = template.format_completion("Rust", "fn main() {}");

        assert!(completion.contains("```Rust"));
        assert!(completion.contains("fn main() {}"));
    }

    #[test]
    fn test_format_completion_without_tags() {
        let template = PromptTemplate::completion_style();
        let completion = template.format_completion("Rust", "fn main() {}");

        assert_eq!(completion, "fn main() {}");
        assert!(!completion.contains("```"));
    }

    #[test]
    fn test_to_example() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);
        let tuple = sample_tuple();

        let example = exporter.to_example(&tuple, "test_001");

        assert_eq!(example.id, "test_001");
        assert_eq!(example.source_language, "python");
        assert_eq!(example.target_language, "rust");
        assert!(example.prompt.contains("def add"));
        assert!(example.completion.contains("fn add"));
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();

        assert_eq!(config.format, ExportFormat::Jsonl);
        assert!((config.train_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert!(config.max_examples.is_none());
    }

    #[test]
    fn test_split_train_val_ratio() {
        let config = ExportConfig {
            train_ratio: 0.8,
            ..Default::default()
        };
        let exporter = EntrenarExporter::new(config);

        let examples: Vec<_> = (0..1000)
            .map(|i| CodeTranslationExample {
                id: format!("ex_{i}"),
                source_language: "Python".to_string(),
                target_language: "Rust".to_string(),
                source_code: format!("x = {i}"),
                target_code: format!("let x = {i};"),
                prompt: String::new(),
                completion: String::new(),
            })
            .collect();

        let (train, val) = exporter.split_train_val(&examples);

        // Should be approximately 80/20 split
        let train_ratio = train.len() as f64 / examples.len() as f64;
        assert!(train_ratio > 0.7 && train_ratio < 0.9);
        assert_eq!(train.len() + val.len(), examples.len());
    }

    #[test]
    fn test_split_deterministic() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);

        let examples: Vec<_> = (0..100)
            .map(|i| CodeTranslationExample {
                id: format!("ex_{i}"),
                source_language: "Python".to_string(),
                target_language: "Rust".to_string(),
                source_code: format!("x = {i}"),
                target_code: format!("let x = {i};"),
                prompt: String::new(),
                completion: String::new(),
            })
            .collect();

        let (train1, _) = exporter.split_train_val(&examples);
        let (train2, _) = exporter.split_train_val(&examples);

        assert_eq!(train1.len(), train2.len());
    }

    #[test]
    fn test_generate_entrenar_config() {
        let config = generate_entrenar_config(
            Path::new("data/train"),
            Path::new("outputs/model"),
            16,
        );

        assert!(config.contains("lora:"));
        assert!(config.contains("rank: 16"));
        assert!(config.contains("alpha: 32"));
        assert!(config.contains("train.jsonl"));
        assert!(config.contains("val.jsonl"));
    }

    #[test]
    fn test_export_format_serde() {
        let json = serde_json::to_string(&ExportFormat::Jsonl).unwrap();
        assert_eq!(json, "\"jsonl\"");

        let parsed: ExportFormat = serde_json::from_str("\"parquet\"").unwrap();
        assert_eq!(parsed, ExportFormat::Parquet);
    }

    // RED PHASE: Tests that require full entrenar integration

    #[test]
    #[ignore = "requires filesystem setup"]
    fn test_export_to_jsonl() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);
        let tuples = vec![sample_tuple()];

        let dir = tempfile::tempdir().unwrap();
        let stats = exporter.export(&tuples, dir.path()).unwrap();

        assert_eq!(stats.total, 1);
        assert!(dir.path().join("train.jsonl").exists() || dir.path().join("val.jsonl").exists());
    }

    #[test]
    #[ignore = "requires entrenar integration"]
    fn test_export_to_parquet() {
        // TODO: Implement Parquet export
        // let config = ExportConfig { format: ExportFormat::Parquet, ..Default::default() };
        // let exporter = EntrenarExporter::new(config);
        // let stats = exporter.export(&tuples, dir.path()).unwrap();
        // assert!(dir.path().join("train.parquet").exists());
        unimplemented!("Parquet export not yet implemented")
    }

    #[test]
    #[ignore = "requires entrenar integration"]
    fn test_load_in_entrenar() {
        // TODO: Verify exported data loads correctly in entrenar
        // let config = entrenar::config::load_config("train_config.yaml").unwrap();
        // assert!(config.data.train.exists());
        unimplemented!("Entrenar integration test not yet implemented")
    }

    #[test]
    #[ignore = "requires LLM evaluation"]
    fn test_prompt_quality() {
        // TODO: Evaluate prompt quality with actual LLM
        // - Measure translation accuracy
        // - Compare different prompt templates
        // - Validate on held-out test set
        unimplemented!("LLM evaluation not yet implemented")
    }
}
