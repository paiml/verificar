//! Verificar CLI - Synthetic Data Factory for Domain-Specific Code Intelligence
//!
//! Generate verified test cases for transpiler validation.

use clap::{Parser, Subcommand};
use std::io::Write;
use verificar::generator::{Generator, SamplingStrategy};
use verificar::Language;

/// Verificar - Synthetic Data Factory for Code Intelligence
#[derive(Parser)]
#[command(name = "verificar")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate test programs
    Generate {
        /// Target language (python, bash, ruby, typescript, rust)
        #[arg(short, long, default_value = "python")]
        language: String,

        /// Number of programs to generate
        #[arg(short, long, default_value = "10")]
        count: usize,

        /// Maximum AST depth
        #[arg(short = 'd', long, default_value = "3")]
        max_depth: usize,

        /// Generation strategy (exhaustive, coverage, random)
        #[arg(short, long, default_value = "exhaustive")]
        strategy: String,

        /// Random seed for reproducible generation
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        output: String,
    },

    /// Show generation statistics
    Stats {
        /// Target language
        #[arg(short, long, default_value = "python")]
        language: String,

        /// Maximum AST depth
        #[arg(short = 'd', long, default_value = "3")]
        max_depth: usize,
    },

    /// Generate coverage-guided test corpus (NAUTILUS-style)
    Corpus {
        /// Target language
        #[arg(short, long, default_value = "python")]
        language: String,

        /// Number of programs to generate
        #[arg(short, long, default_value = "100")]
        count: usize,

        /// Maximum AST depth
        #[arg(short = 'd', long, default_value = "3")]
        max_depth: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "python" | "py" => Language::Python,
        "bash" | "sh" => Language::Bash,
        "ruby" | "rb" => Language::Ruby,
        "typescript" | "ts" => Language::TypeScript,
        "rust" | "rs" => Language::Rust,
        _ => {
            eprintln!("Warning: Unknown language '{s}', defaulting to Python");
            Language::Python
        }
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            language,
            count,
            max_depth,
            strategy,
            seed,
            output,
        } => {
            let lang = parse_language(&language);
            let generator = Generator::new(lang);

            let programs = match strategy.as_str() {
                "exhaustive" => generator.generate_exhaustive(max_depth),
                "coverage" => generator.generate_coverage_guided(count, max_depth, seed),
                "random" => {
                    let strat = SamplingStrategy::Random { seed, count };
                    generator.generate(strat, count).unwrap_or_else(|e| {
                        eprintln!("Generation error: {e}");
                        vec![]
                    })
                }
                _ => {
                    eprintln!("Unknown strategy '{strategy}', using exhaustive");
                    generator.generate_exhaustive(max_depth)
                }
            };

            // Limit to requested count for exhaustive
            let programs: Vec<_> = programs.into_iter().take(count).collect();

            match output.as_str() {
                "json" => {
                    let items: Vec<_> = programs
                        .iter()
                        .map(|p| {
                            serde_json::json!({
                                "code": p.code,
                                "language": p.language.to_string(),
                                "ast_depth": p.ast_depth,
                                "features": p.features
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&items).unwrap_or_default()
                    );
                }
                _ => {
                    for (i, prog) in programs.iter().enumerate() {
                        println!("--- Program {} (depth: {}) ---", i + 1, prog.ast_depth);
                        println!("{}", prog.code);
                        println!();
                    }
                }
            }
        }

        Commands::Stats {
            language,
            max_depth,
        } => {
            let lang = parse_language(&language);
            let generator = Generator::new(lang);
            let stats = generator.generate_with_stats(max_depth);

            println!("Generation Statistics for {language} (depth {max_depth}):");
            println!("  Total generated: {}", stats.total_generated);
            println!("  Valid programs:  {}", stats.valid_count);
            println!("  Invalid:         {}", stats.invalid_count);
            println!("  Pass rate:       {:.1}%", stats.pass_rate());
        }

        Commands::Corpus {
            language,
            count,
            max_depth,
            seed,
        } => {
            let lang = parse_language(&language);
            let generator = Generator::new(lang);
            let (programs, stats) =
                generator.generate_coverage_guided_with_map(count, max_depth, seed, None);

            println!("Coverage-Guided Corpus Generation (NAUTILUS-style)");
            println!("================================================");
            println!("Language:           {language}");
            println!("Max depth:          {max_depth}");
            println!("Seed:               {seed}");
            println!();
            println!("Coverage Statistics:");
            println!("  Corpus size:      {}", stats.corpus_size);
            println!("  Node types:       {}", stats.node_types_covered);
            println!("  AST paths:        {}", stats.ast_paths_covered);
            println!("  Features:         {}", stats.features_covered);
            println!();
            println!("Generated {} programs:", programs.len());
            println!();

            // Write programs to stdout
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            for (i, prog) in programs.iter().enumerate() {
                writeln!(handle, "# Program {}", i + 1).ok();
                writeln!(handle, "{}", prog.code).ok();
                writeln!(handle).ok();
            }
        }
    }
}
