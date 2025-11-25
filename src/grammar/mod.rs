//! Grammar definitions for source languages
//!
//! This module provides grammar definitions for each supported source language.
//! Grammars can be defined using tree-sitter or pest PEGs.
//!
//! # Supported Languages
//!
//! - Python (for depyler)
//! - Bash (for bashrs)
//! - Ruby (for ruchy)
//! - TypeScript (for decy)

mod bash;
mod python;

pub use bash::BashGrammar;
pub use python::PythonGrammar;

use crate::Language;

/// Trait for language grammar definitions
pub trait Grammar: Send + Sync + std::fmt::Debug {
    /// Get the language this grammar defines
    fn language(&self) -> Language;

    /// Validate that a code string conforms to the grammar
    fn validate(&self, code: &str) -> bool;

    /// Get the maximum depth for enumeration
    fn max_enumeration_depth(&self) -> usize {
        5 // Default per spec - combinatorial explosion at depth 5-6
    }
}

/// Create a grammar for the specified language
#[must_use]
pub fn grammar_for(language: Language) -> Box<dyn Grammar> {
    match language {
        Language::Python => Box::new(PythonGrammar::new()),
        Language::Bash => Box::new(BashGrammar::new()),
        Language::Ruby => todo!("Ruby grammar not yet implemented"),
        Language::TypeScript => todo!("TypeScript grammar not yet implemented"),
        Language::Rust => todo!("Rust grammar not yet implemented"),
    }
}
