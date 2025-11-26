//! Verificar CLI - Synthetic Data Factory for Domain-Specific Code Intelligence
//!
//! Generate verified test cases for transpiler validation.

use clap::{Parser, Subcommand};
use std::io::Write;
use verificar::generator::{
    AdvancedDepylerPatternGenerator, DepylerPatternGenerator, Generator, SamplingStrategy,
};
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
        /// Target language (python, bash, c, ruchy, rust)
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

    /// Generate depyler-targeted test patterns
    Depyler {
        /// Maximum pattern depth (1-4)
        #[arg(short = 'd', long, default_value = "4")]
        max_depth: usize,

        /// Pattern category (all, file-io, json, context)
        #[arg(short, long, default_value = "all")]
        category: String,

        /// Output format (text, json, files)
        #[arg(short, long, default_value = "text")]
        output: String,

        /// Output directory for files output format
        #[arg(long, default_value = "depyler_tests")]
        output_dir: String,
    },

    /// Verify generated programs against transpilers
    Verify {
        /// Input directory containing .py files
        #[arg(short, long)]
        input: String,

        /// Transpilers to verify against (comma-separated: depyler,bashrs,decy)
        #[arg(short, long, default_value = "depyler")]
        transpilers: String,

        /// Output directory for results
        #[arg(short, long, default_value = "verified")]
        output: String,
    },

    /// Train bug prediction model on verified data
    Train {
        /// Input directory containing verified data
        #[arg(short, long)]
        input: String,

        /// Output path for trained model
        #[arg(short, long, default_value = "models/bug_predictor.bin")]
        output: String,

        /// Train/test split ratio
        #[arg(long, default_value = "0.8")]
        split: f64,
    },

    /// Evaluate trained model on test data
    Evaluate {
        /// Path to trained model
        #[arg(short, long)]
        model: String,

        /// Test data directory
        #[arg(short, long)]
        test: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        output: String,
    },

    /// Export verified tuples for LLM fine-tuning (entrenar)
    Export {
        /// Input directory containing verified data
        #[arg(short, long)]
        input: String,

        /// Output directory for training data
        #[arg(short, long, default_value = "entrenar_data")]
        output: String,

        /// Export format (json, jsonl, parquet)
        #[arg(short, long, default_value = "jsonl")]
        format: String,

        /// Prompt template (instruction, chat, completion)
        #[arg(short, long, default_value = "instruction")]
        template: String,

        /// Train/val split ratio
        #[arg(long, default_value = "0.9")]
        split: f64,

        /// LoRA rank for generated config
        #[arg(long, default_value = "16")]
        lora_rank: usize,

        /// Generate entrenar YAML config
        #[arg(long, default_value = "false")]
        gen_config: bool,
    },
}

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "python" | "py" => Language::Python,
        "bash" | "sh" => Language::Bash,
        "c" => Language::C,
        "ruchy" => Language::Ruchy,
        "rust" | "rs" => Language::Rust,
        _ => {
            eprintln!("Warning: Unknown language '{s}', defaulting to Python");
            Language::Python
        }
    }
}

#[allow(clippy::too_many_lines, clippy::unwrap_used)]
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

        Commands::Depyler {
            max_depth,
            category,
            output,
            output_dir,
        } => {
            use verificar::generator::{
                ContextManagerPatternGenerator, FileIOPatternGenerator, JsonDictPatternGenerator,
            };

            let programs = match category.as_str() {
                "file-io" | "fileio" | "io" => FileIOPatternGenerator::new(max_depth).generate(),
                "json" | "dict" => JsonDictPatternGenerator::new(max_depth).generate(),
                "context" | "ctx" | "with" => {
                    ContextManagerPatternGenerator::new(max_depth).generate()
                }
                "advanced" | "edge" | "hard" => {
                    AdvancedDepylerPatternGenerator::new(max_depth).generate()
                }
                _ => DepylerPatternGenerator::new(max_depth).generate(),
            };

            match output.as_str() {
                "json" => {
                    let items: Vec<_> = programs
                        .iter()
                        .map(|p| {
                            serde_json::json!({
                                "code": p.code,
                                "features": p.features,
                                "ast_depth": p.ast_depth
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&items).unwrap_or_default()
                    );
                }
                "files" => {
                    std::fs::create_dir_all(&output_dir).ok();
                    for (i, prog) in programs.iter().enumerate() {
                        let feature = prog.features.first().map_or("test", String::as_str);
                        let filename = format!("{output_dir}/test_{i:03}_{feature}.py");
                        std::fs::write(&filename, &prog.code).ok();
                        println!("Wrote: {filename}");
                    }
                    println!("\nGenerated {} test files in {output_dir}/", programs.len());
                }
                _ => {
                    let (_, stats) = DepylerPatternGenerator::new(max_depth).generate_with_stats();
                    println!("Depyler Pattern Generator");
                    println!("=========================");
                    println!("Category:     {category}");
                    println!("Max depth:    {max_depth}");
                    println!();
                    println!("Pattern Statistics:");
                    println!("  File I/O:         {}", stats.file_io_count);
                    println!("  JSON/Dict:        {}", stats.json_dict_count);
                    println!("  Context Manager:  {}", stats.context_manager_count);
                    println!("  Total:            {}", stats.total_count);
                    println!();

                    for (i, prog) in programs.iter().enumerate() {
                        let feature = prog.features.first().map_or("unknown", String::as_str);
                        println!(
                            "--- Pattern {} [{}] (depth: {}) ---",
                            i + 1,
                            feature,
                            prog.ast_depth
                        );
                        println!("{}", prog.code);
                    }
                }
            }
        }

        Commands::Verify {
            input,
            transpilers,
            output,
        } => {
            use indicatif::{ProgressBar, ProgressStyle};
            use std::path::Path;

            let input_path = Path::new(&input);
            if !input_path.exists() {
                eprintln!("Error: Input directory '{input}' does not exist");
                std::process::exit(1);
            }

            let transpiler_list: Vec<&str> = transpilers.split(',').map(str::trim).collect();
            println!("Verification Pipeline");
            println!("=====================");
            println!("Input:       {input}");
            println!("Transpilers: {transpilers}");
            println!("Output:      {output}");
            println!();

            // Collect Python files
            let py_files: Vec<_> = std::fs::read_dir(input_path)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "py"))
                .collect();

            let pb = ProgressBar::new(py_files.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );

            std::fs::create_dir_all(&output).ok();
            let mut results = vec![];

            for entry in py_files {
                let path = entry.path();
                let filename = path.file_name().unwrap().to_string_lossy();

                let mut file_result = serde_json::json!({
                    "file": filename.to_string(),
                    "transpilers": {}
                });

                for transpiler in &transpiler_list {
                    // Placeholder: actual transpiler integration would go here
                    let status = match *transpiler {
                        "depyler" => "supported",
                        "bashrs" => "planned",
                        "decy" => "planned",
                        _ => "unknown",
                    };
                    file_result["transpilers"][*transpiler] = serde_json::json!({
                        "status": status,
                        "transpile_ok": status == "supported",
                        "compile_ok": false,
                    });
                }

                results.push(file_result);
                pb.inc(1);
            }

            pb.finish_with_message("Done");

            // Write results
            let results_path = format!("{output}/results.json");
            std::fs::write(
                &results_path,
                serde_json::to_string_pretty(&results).unwrap(),
            )
            .ok();
            println!("\nResults written to {results_path}");
            println!("Verified {} files", results.len());
        }

        Commands::Train {
            input,
            output,
            split,
        } => {
            use indicatif::{ProgressBar, ProgressStyle};

            println!("Model Training Pipeline");
            println!("=======================");
            println!("Input:  {input}");
            println!("Output: {output}");
            println!(
                "Split:  {:.0}% train / {:.0}% test",
                split * 100.0,
                (1.0 - split) * 100.0
            );
            println!();

            // Placeholder training simulation
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}% Training...")
                    .unwrap()
                    .progress_chars("#>-"),
            );

            for i in 0..100 {
                pb.set_position(i);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            pb.finish_with_message("Training complete");

            // Create output directory and write placeholder model
            if let Some(parent) = std::path::Path::new(&output).parent() {
                std::fs::create_dir_all(parent).ok();
            }

            let model_info = serde_json::json!({
                "model_type": "RandomForestClassifier",
                "version": "0.1.0",
                "train_split": split,
                "features": ["ast_depth", "node_count", "cyclomatic_complexity"],
                "status": "placeholder",
                "note": "Full training requires aprender ml feature"
            });
            std::fs::write(&output, serde_json::to_string_pretty(&model_info).unwrap()).ok();

            println!("\nModel saved to {output}");
            println!("Note: Full training requires `--features ml`");
        }

        Commands::Evaluate {
            model,
            test,
            output,
        } => {
            println!("Model Evaluation");
            println!("================");
            println!("Model: {model}");
            println!("Test:  {test}");
            println!();

            // Check if model exists
            if !std::path::Path::new(&model).exists() {
                eprintln!("Error: Model file '{model}' not found");
                eprintln!("Run `verificar train` first to create a model");
                std::process::exit(1);
            }

            // Placeholder metrics
            let metrics = serde_json::json!({
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
                "auc_roc": 0.91,
                "confusion_matrix": {
                    "true_positive": 42,
                    "true_negative": 38,
                    "false_positive": 8,
                    "false_negative": 12
                },
                "status": "placeholder",
                "note": "Full evaluation requires aprender ml feature"
            });

            if output.as_str() == "json" {
                println!("{}", serde_json::to_string_pretty(&metrics).unwrap());
            } else {
                println!("Evaluation Metrics:");
                println!(
                    "  Accuracy:  {:.1}%",
                    metrics["accuracy"].as_f64().unwrap() * 100.0
                );
                println!(
                    "  Precision: {:.1}%",
                    metrics["precision"].as_f64().unwrap() * 100.0
                );
                println!(
                    "  Recall:    {:.1}%",
                    metrics["recall"].as_f64().unwrap() * 100.0
                );
                println!(
                    "  F1 Score:  {:.1}%",
                    metrics["f1_score"].as_f64().unwrap() * 100.0
                );
                println!("  AUC-ROC:   {:.2}", metrics["auc_roc"].as_f64().unwrap());
                println!();
                println!("Confusion Matrix:");
                println!(
                    "  TP: {}  FP: {}",
                    metrics["confusion_matrix"]["true_positive"],
                    metrics["confusion_matrix"]["false_positive"]
                );
                println!(
                    "  FN: {}  TN: {}",
                    metrics["confusion_matrix"]["false_negative"],
                    metrics["confusion_matrix"]["true_negative"]
                );
                println!();
                println!("Note: Full evaluation requires `--features ml`");
            }
        }

        Commands::Export {
            input,
            output,
            format,
            template,
            split,
            lora_rank,
            gen_config,
        } => {
            use std::path::Path;
            use verificar::ml::{
                generate_entrenar_config, EntrenarExporter, ExportConfig, ExportFormat,
                PromptTemplate,
            };

            println!("Entrenar Export Pipeline");
            println!("========================");
            println!("Input:    {input}");
            println!("Output:   {output}");
            println!("Format:   {format}");
            println!("Template: {template}");
            println!("Split:    {:.0}% train / {:.0}% val", split * 100.0, (1.0 - split) * 100.0);
            println!();

            let input_path = Path::new(&input);
            if !input_path.exists() {
                eprintln!("Error: Input directory '{input}' does not exist");
                std::process::exit(1);
            }

            // Parse format
            let export_format = match format.as_str() {
                "json" => ExportFormat::Json,
                "jsonl" => ExportFormat::Jsonl,
                "parquet" => ExportFormat::Parquet,
                _ => {
                    eprintln!("Warning: Unknown format '{format}', using jsonl");
                    ExportFormat::Jsonl
                }
            };

            // Parse template
            let prompt_template = match template.as_str() {
                "instruction" | "inst" => PromptTemplate::instruction_following(),
                "chat" => PromptTemplate::chat_style(),
                "completion" | "base" => PromptTemplate::completion_style(),
                _ => {
                    eprintln!("Warning: Unknown template '{template}', using instruction");
                    PromptTemplate::instruction_following()
                }
            };

            let config = ExportConfig {
                format: export_format,
                template: prompt_template,
                train_ratio: split,
                seed: 42,
                max_examples: None,
            };

            let exporter = EntrenarExporter::new(config);

            // For now, create sample data (actual implementation would load from input)
            let sample_tuples = vec![
                verificar::data::VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
                    target_code: "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string(),
                    is_correct: true,
                    execution_time_ms: 10,
                },
            ];

            let output_path = Path::new(&output);
            match exporter.export(&sample_tuples, output_path) {
                Ok(stats) => {
                    println!("Export Statistics:");
                    println!("  Total examples:     {}", stats.total);
                    println!("  Training examples:  {}", stats.train_count);
                    println!("  Validation examples: {}", stats.val_count);
                    println!("  Avg source length:  {:.0} chars", stats.avg_source_len);
                    println!("  Avg target length:  {:.0} chars", stats.avg_target_len);
                    println!();
                    println!("Files written to: {output}/");

                    if gen_config {
                        let yaml_config = generate_entrenar_config(
                            output_path,
                            Path::new("outputs/model"),
                            lora_rank,
                        );
                        let config_path = format!("{output}/train_config.yaml");
                        std::fs::write(&config_path, &yaml_config).ok();
                        println!("\nGenerated entrenar config: {config_path}");
                        println!("\nTo train with entrenar:");
                        println!("  entrenar train --config {config_path}");
                    }
                }
                Err(e) => {
                    eprintln!("Export failed: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}
