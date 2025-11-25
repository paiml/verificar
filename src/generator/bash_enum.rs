//! Bash exhaustive enumeration
//!
//! Generates all valid Bash programs up to a specified AST depth.
//! Uses a simplified Bash grammar for combinatorial generation.

#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::format_collect)]
#![allow(clippy::inefficient_to_string)]

use super::GeneratedCode;
use crate::Language;

/// Bash AST node types for generation
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum BashNode {
    /// Script (root node with optional shebang)
    Script {
        /// Optional shebang line
        shebang: Option<String>,
        /// Script body statements
        body: Vec<BashNode>,
    },
    /// Variable assignment: `name=value`
    Assignment {
        /// Variable name
        name: String,
        /// Value expression
        value: Box<BashNode>,
    },
    /// Command execution
    Command {
        /// Command name
        name: String,
        /// Command arguments
        args: Vec<BashNode>,
    },
    /// If statement
    If {
        /// Condition (test command)
        condition: Box<BashNode>,
        /// Then body
        then_body: Vec<BashNode>,
        /// Optional else body
        else_body: Vec<BashNode>,
    },
    /// For loop
    For {
        /// Loop variable name
        var: String,
        /// Iterable (words)
        items: Vec<BashNode>,
        /// Loop body
        body: Vec<BashNode>,
    },
    /// While loop
    While {
        /// Condition
        condition: Box<BashNode>,
        /// Loop body
        body: Vec<BashNode>,
    },
    /// Function definition
    Function {
        /// Function name
        name: String,
        /// Function body
        body: Vec<BashNode>,
    },
    /// Case statement
    Case {
        /// Value to match
        value: Box<BashNode>,
        /// Pattern-body pairs
        patterns: Vec<(String, Vec<BashNode>)>,
    },
    /// Test expression: `[ expr ]` or `[[ expr ]]`
    Test {
        /// Use double brackets `[[`
        double: bool,
        /// Test expression
        expr: Box<BashNode>,
    },
    /// Binary comparison: `left op right`
    Compare {
        /// Left operand
        left: Box<BashNode>,
        /// Comparison operator
        op: BashCompareOp,
        /// Right operand
        right: Box<BashNode>,
    },
    /// Arithmetic expression: `$((expr))`
    Arithmetic(Box<BashNode>),
    /// Binary arithmetic operation
    ArithOp {
        /// Left operand
        left: Box<BashNode>,
        /// Operator
        op: BashArithOp,
        /// Right operand
        right: Box<BashNode>,
    },
    /// Variable reference: `$name` or `${name}`
    Variable(String),
    /// String literal
    StringLit(String),
    /// Integer literal
    IntLit(i64),
    /// Array literal
    Array(Vec<BashNode>),
    /// Pipe: `cmd1 | cmd2`
    Pipe {
        /// Left command
        left: Box<BashNode>,
        /// Right command
        right: Box<BashNode>,
    },
    /// Command substitution: `$(cmd)`
    CommandSubst(Box<BashNode>),
    /// Redirection
    Redirect {
        /// Command
        command: Box<BashNode>,
        /// Redirect type
        redirect_type: RedirectType,
        /// Target file/fd
        target: String,
    },
}

/// Bash comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashCompareOp {
    /// Numeric equal `-eq`
    NumEq,
    /// Numeric not equal `-ne`
    NumNe,
    /// Numeric less than `-lt`
    NumLt,
    /// Numeric greater than `-gt`
    NumGt,
    /// Numeric less than or equal `-le`
    NumLe,
    /// Numeric greater than or equal `-ge`
    NumGe,
    /// String equal `=` or `==`
    StrEq,
    /// String not equal `!=`
    StrNe,
    /// String less than `<`
    StrLt,
    /// String greater than `>`
    StrGt,
}

impl BashCompareOp {
    /// Get all comparison operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::NumEq,
            Self::NumNe,
            Self::NumLt,
            Self::NumGt,
            Self::NumLe,
            Self::NumGe,
            Self::StrEq,
            Self::StrNe,
        ]
    }

    /// Convert to Bash operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::NumEq => "-eq",
            Self::NumNe => "-ne",
            Self::NumLt => "-lt",
            Self::NumGt => "-gt",
            Self::NumLe => "-le",
            Self::NumGe => "-ge",
            Self::StrEq => "==",
            Self::StrNe => "!=",
            Self::StrLt => "<",
            Self::StrGt => ">",
        }
    }
}

/// Bash arithmetic operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashArithOp {
    /// Addition `+`
    Add,
    /// Subtraction `-`
    Sub,
    /// Multiplication `*`
    Mult,
    /// Division `/`
    Div,
    /// Modulo `%`
    Mod,
}

impl BashArithOp {
    /// Get all arithmetic operators
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[Self::Add, Self::Sub, Self::Mult, Self::Div, Self::Mod]
    }

    /// Convert to Bash operator string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mult => "*",
            Self::Div => "/",
            Self::Mod => "%",
        }
    }
}

/// Redirect types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedirectType {
    /// Output `>`
    Output,
    /// Append `>>`
    Append,
    /// Input `<`
    Input,
    /// Stderr to stdout `2>&1`
    StderrToStdout,
}

impl RedirectType {
    /// Convert to Bash redirect string
    #[must_use]
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Output => ">",
            Self::Append => ">>",
            Self::Input => "<",
            Self::StderrToStdout => "2>&1",
        }
    }
}

impl BashNode {
    /// Convert AST node to Bash code string
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_code(&self) -> String {
        match self {
            Self::Script { shebang, body } => {
                let mut code = String::new();
                if let Some(s) = shebang {
                    code.push_str(s);
                    code.push('\n');
                }
                for stmt in body {
                    code.push_str(&stmt.to_code());
                    code.push('\n');
                }
                code.trim_end().to_string()
            }
            Self::Assignment { name, value } => {
                format!("{}={}", name, value.to_code())
            }
            Self::Command { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str: Vec<String> = args.iter().map(|a| a.to_code()).collect();
                    format!("{} {}", name, args_str.join(" "))
                }
            }
            Self::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond = condition.to_code();
                let then_str: String = then_body
                    .iter()
                    .map(|s| format!("    {}", s.to_code()))
                    .collect::<Vec<_>>()
                    .join("\n");

                if else_body.is_empty() {
                    format!("if {}; then\n{}\nfi", cond, then_str)
                } else {
                    let else_str: String = else_body
                        .iter()
                        .map(|s| format!("    {}", s.to_code()))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("if {}; then\n{}\nelse\n{}\nfi", cond, then_str, else_str)
                }
            }
            Self::For { var, items, body } => {
                let items_str: Vec<String> = items.iter().map(|i| i.to_code()).collect();
                let body_str: String = body
                    .iter()
                    .map(|s| format!("    {}", s.to_code()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "for {} in {}; do\n{}\ndone",
                    var,
                    items_str.join(" "),
                    body_str
                )
            }
            Self::While { condition, body } => {
                let body_str: String = body
                    .iter()
                    .map(|s| format!("    {}", s.to_code()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("while {}; do\n{}\ndone", condition.to_code(), body_str)
            }
            Self::Function { name, body } => {
                let body_str: String = body
                    .iter()
                    .map(|s| format!("    {}", s.to_code()))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("{}() {{\n{}\n}}", name, body_str)
            }
            Self::Case { value, patterns } => {
                let patterns_str: String = patterns
                    .iter()
                    .map(|(pat, body)| {
                        let body_str: String = body
                            .iter()
                            .map(|s| s.to_code())
                            .collect::<Vec<_>>()
                            .join("; ");
                        format!("    {}) {};;\n", pat, body_str)
                    })
                    .collect();
                format!("case {} in\n{}esac", value.to_code(), patterns_str)
            }
            Self::Test { double, expr } => {
                if *double {
                    format!("[[ {} ]]", expr.to_code())
                } else {
                    format!("[ {} ]", expr.to_code())
                }
            }
            Self::Compare { left, op, right } => {
                format!("{} {} {}", left.to_code(), op.to_str(), right.to_code())
            }
            Self::Arithmetic(expr) => {
                format!("$(({}))", expr.to_code())
            }
            Self::ArithOp { left, op, right } => {
                format!("{} {} {}", left.to_code(), op.to_str(), right.to_code())
            }
            Self::Variable(name) => format!("${}", name),
            Self::StringLit(s) => {
                if s.contains(' ') || s.contains('$') {
                    format!("\"{}\"", s)
                } else {
                    s.clone()
                }
            }
            Self::IntLit(n) => n.to_string(),
            Self::Array(items) => {
                let items_str: Vec<String> = items.iter().map(|i| i.to_code()).collect();
                format!("({})", items_str.join(" "))
            }
            Self::Pipe { left, right } => {
                format!("{} | {}", left.to_code(), right.to_code())
            }
            Self::CommandSubst(cmd) => {
                format!("$({})", cmd.to_code())
            }
            Self::Redirect {
                command,
                redirect_type,
                target,
            } => {
                format!(
                    "{} {} {}",
                    command.to_code(),
                    redirect_type.to_str(),
                    target
                )
            }
        }
    }

    /// Calculate AST depth
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Script { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::Assignment { value, .. } => 1 + value.depth(),
            Self::Command { args, .. } => 1 + args.iter().map(Self::depth).max().unwrap_or(0),
            Self::If {
                condition,
                then_body,
                else_body,
            } => {
                let cond_depth = condition.depth();
                let then_depth = then_body.iter().map(Self::depth).max().unwrap_or(0);
                let else_depth = else_body.iter().map(Self::depth).max().unwrap_or(0);
                1 + cond_depth.max(then_depth).max(else_depth)
            }
            Self::For { items, body, .. } => {
                let items_depth = items.iter().map(Self::depth).max().unwrap_or(0);
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + items_depth.max(body_depth)
            }
            Self::While { condition, body } => {
                let cond_depth = condition.depth();
                let body_depth = body.iter().map(Self::depth).max().unwrap_or(0);
                1 + cond_depth.max(body_depth)
            }
            Self::Function { body, .. } => 1 + body.iter().map(Self::depth).max().unwrap_or(0),
            Self::Case { value, patterns } => {
                let val_depth = value.depth();
                let pat_depth = patterns
                    .iter()
                    .flat_map(|(_, body)| body.iter().map(Self::depth))
                    .max()
                    .unwrap_or(0);
                1 + val_depth.max(pat_depth)
            }
            Self::Test { expr, .. } => 1 + expr.depth(),
            Self::Compare { left, right, .. } => 1 + left.depth().max(right.depth()),
            Self::Arithmetic(expr) => 1 + expr.depth(),
            Self::ArithOp { left, right, .. } => 1 + left.depth().max(right.depth()),
            Self::Pipe { left, right } => 1 + left.depth().max(right.depth()),
            Self::CommandSubst(cmd) => 1 + cmd.depth(),
            Self::Redirect { command, .. } => 1 + command.depth(),
            Self::Variable(_) | Self::StringLit(_) | Self::IntLit(_) => 1,
            Self::Array(items) => 1 + items.iter().map(Self::depth).max().unwrap_or(0),
        }
    }

    /// Extract features used in this node
    #[must_use]
    pub fn features(&self) -> Vec<String> {
        let mut features = Vec::new();

        match self {
            Self::Script { body, .. } => {
                features.push("script".to_string());
                for stmt in body {
                    features.extend(stmt.features());
                }
            }
            Self::Assignment { .. } => features.push("assignment".to_string()),
            Self::Command { name, .. } => {
                features.push("command".to_string());
                features.push(format!("cmd_{}", name));
            }
            Self::If { .. } => features.push("if".to_string()),
            Self::For { .. } => features.push("for".to_string()),
            Self::While { .. } => features.push("while".to_string()),
            Self::Function { .. } => features.push("function".to_string()),
            Self::Case { .. } => features.push("case".to_string()),
            Self::Test { double, .. } => {
                features.push(
                    if *double {
                        "test_double"
                    } else {
                        "test_single"
                    }
                    .to_string(),
                );
            }
            Self::Compare { op, .. } => features.push(format!("compare_{}", op.to_str())),
            Self::Arithmetic(_) => features.push("arithmetic".to_string()),
            Self::ArithOp { op, .. } => features.push(format!("arith_{}", op.to_str())),
            Self::Variable(_) => features.push("variable".to_string()),
            Self::StringLit(_) => features.push("string".to_string()),
            Self::IntLit(_) => features.push("integer".to_string()),
            Self::Array(_) => features.push("array".to_string()),
            Self::Pipe { .. } => features.push("pipe".to_string()),
            Self::CommandSubst(_) => features.push("command_subst".to_string()),
            Self::Redirect { redirect_type, .. } => {
                features.push(format!("redirect_{}", redirect_type.to_str()));
            }
        }

        features
    }
}

/// Bash program enumerator
#[derive(Debug)]
pub struct BashEnumerator {
    /// Maximum AST depth
    max_depth: usize,
}

impl BashEnumerator {
    /// Create a new Bash enumerator
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Enumerate all Bash programs up to the configured depth
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn enumerate_programs(&self) -> Vec<GeneratedCode> {
        let mut programs = Vec::new();

        // Generate simple assignments
        for var in &["x", "y", "result"] {
            for val in [1, 0, 42] {
                let node = BashNode::Assignment {
                    name: var.to_string(),
                    value: Box::new(BashNode::IntLit(val)),
                };
                if node.depth() <= self.max_depth {
                    programs.push(self.node_to_generated(&node));
                }
            }
            // String assignments
            for s in &["hello", "world"] {
                let node = BashNode::Assignment {
                    name: var.to_string(),
                    value: Box::new(BashNode::StringLit(s.to_string())),
                };
                if node.depth() <= self.max_depth {
                    programs.push(self.node_to_generated(&node));
                }
            }
        }

        // Generate echo commands
        for arg in &["$x", "hello", "$HOME"] {
            let node = BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::StringLit(arg.to_string())],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate if statements (depth 2+)
        if self.max_depth >= 2 {
            for var in &["x", "y"] {
                for op in &[
                    BashCompareOp::NumEq,
                    BashCompareOp::NumGt,
                    BashCompareOp::NumLt,
                ] {
                    let node = BashNode::If {
                        condition: Box::new(BashNode::Test {
                            double: false,
                            expr: Box::new(BashNode::Compare {
                                left: Box::new(BashNode::Variable(var.to_string())),
                                op: *op,
                                right: Box::new(BashNode::IntLit(0)),
                            }),
                        }),
                        then_body: vec![BashNode::Command {
                            name: "echo".to_string(),
                            args: vec![BashNode::StringLit("yes".to_string())],
                        }],
                        else_body: vec![],
                    };
                    if node.depth() <= self.max_depth {
                        programs.push(self.node_to_generated(&node));
                    }
                }
            }
        }

        // Generate for loops (depth 2+)
        if self.max_depth >= 2 {
            let node = BashNode::For {
                var: "i".to_string(),
                items: vec![
                    BashNode::IntLit(1),
                    BashNode::IntLit(2),
                    BashNode::IntLit(3),
                ],
                body: vec![BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::Variable("i".to_string())],
                }],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate while loops (depth 2+)
        if self.max_depth >= 2 {
            let node = BashNode::While {
                condition: Box::new(BashNode::Test {
                    double: false,
                    expr: Box::new(BashNode::Compare {
                        left: Box::new(BashNode::Variable("x".to_string())),
                        op: BashCompareOp::NumGt,
                        right: Box::new(BashNode::IntLit(0)),
                    }),
                }),
                body: vec![BashNode::Assignment {
                    name: "x".to_string(),
                    value: Box::new(BashNode::Arithmetic(Box::new(BashNode::ArithOp {
                        left: Box::new(BashNode::Variable("x".to_string())),
                        op: BashArithOp::Sub,
                        right: Box::new(BashNode::IntLit(1)),
                    }))),
                }],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate functions (depth 2+)
        if self.max_depth >= 2 {
            for name in &["greet", "main"] {
                let node = BashNode::Function {
                    name: name.to_string(),
                    body: vec![BashNode::Command {
                        name: "echo".to_string(),
                        args: vec![BashNode::StringLit("hello".to_string())],
                    }],
                };
                if node.depth() <= self.max_depth {
                    programs.push(self.node_to_generated(&node));
                }
            }
        }

        // Generate arithmetic expressions
        for op in BashArithOp::all() {
            let node = BashNode::Assignment {
                name: "result".to_string(),
                value: Box::new(BashNode::Arithmetic(Box::new(BashNode::ArithOp {
                    left: Box::new(BashNode::IntLit(1)),
                    op: *op,
                    right: Box::new(BashNode::IntLit(2)),
                }))),
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate pipes (depth 2+)
        if self.max_depth >= 2 {
            let node = BashNode::Pipe {
                left: Box::new(BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::StringLit("hello".to_string())],
                }),
                right: Box::new(BashNode::Command {
                    name: "wc".to_string(),
                    args: vec![BashNode::StringLit("-c".to_string())],
                }),
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate array assignments
        let node = BashNode::Assignment {
            name: "arr".to_string(),
            value: Box::new(BashNode::Array(vec![
                BashNode::IntLit(1),
                BashNode::IntLit(2),
                BashNode::IntLit(3),
            ])),
        };
        if node.depth() <= self.max_depth {
            programs.push(self.node_to_generated(&node));
        }

        // Generate command substitution
        let node = BashNode::Assignment {
            name: "output".to_string(),
            value: Box::new(BashNode::CommandSubst(Box::new(BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::StringLit("hello".to_string())],
            }))),
        };
        if node.depth() <= self.max_depth {
            programs.push(self.node_to_generated(&node));
        }

        // Generate redirections
        for redirect_type in &[RedirectType::Output, RedirectType::Append, RedirectType::Input] {
            let target = match redirect_type {
                RedirectType::Input => "/dev/null",
                _ => "/tmp/output.txt",
            };
            let node = BashNode::Redirect {
                command: Box::new(BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::StringLit("test".to_string())],
                }),
                redirect_type: *redirect_type,
                target: target.to_string(),
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate more commands (cat, test, true, false)
        for cmd in &["cat", "test", "true", "false", "pwd", "ls"] {
            let node = BashNode::Command {
                name: cmd.to_string(),
                args: vec![],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate double bracket tests (depth 2+)
        if self.max_depth >= 2 {
            for var in &["x", "str"] {
                let node = BashNode::If {
                    condition: Box::new(BashNode::Test {
                        double: true,
                        expr: Box::new(BashNode::Compare {
                            left: Box::new(BashNode::Variable(var.to_string())),
                            op: BashCompareOp::StrEq,
                            right: Box::new(BashNode::StringLit("hello".to_string())),
                        }),
                    }),
                    then_body: vec![BashNode::Command {
                        name: "echo".to_string(),
                        args: vec![BashNode::StringLit("match".to_string())],
                    }],
                    else_body: vec![],
                };
                if node.depth() <= self.max_depth {
                    programs.push(self.node_to_generated(&node));
                }
            }
        }

        // Generate if-else statements (depth 2+)
        if self.max_depth >= 2 {
            let node = BashNode::If {
                condition: Box::new(BashNode::Test {
                    double: false,
                    expr: Box::new(BashNode::Compare {
                        left: Box::new(BashNode::Variable("x".to_string())),
                        op: BashCompareOp::NumEq,
                        right: Box::new(BashNode::IntLit(1)),
                    }),
                }),
                then_body: vec![BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::StringLit("one".to_string())],
                }],
                else_body: vec![BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::StringLit("not one".to_string())],
                }],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate case statements (depth 2+)
        if self.max_depth >= 2 {
            let node = BashNode::Case {
                value: Box::new(BashNode::Variable("x".to_string())),
                patterns: vec![
                    (
                        "1".to_string(),
                        vec![BashNode::Command {
                            name: "echo".to_string(),
                            args: vec![BashNode::StringLit("one".to_string())],
                        }],
                    ),
                    (
                        "2".to_string(),
                        vec![BashNode::Command {
                            name: "echo".to_string(),
                            args: vec![BashNode::StringLit("two".to_string())],
                        }],
                    ),
                    (
                        "*".to_string(),
                        vec![BashNode::Command {
                            name: "echo".to_string(),
                            args: vec![BashNode::StringLit("other".to_string())],
                        }],
                    ),
                ],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate string comparison operators
        if self.max_depth >= 2 {
            for op in &[BashCompareOp::StrEq, BashCompareOp::StrNe] {
                let node = BashNode::If {
                    condition: Box::new(BashNode::Test {
                        double: true,
                        expr: Box::new(BashNode::Compare {
                            left: Box::new(BashNode::Variable("str".to_string())),
                            op: *op,
                            right: Box::new(BashNode::StringLit("test".to_string())),
                        }),
                    }),
                    then_body: vec![BashNode::Command {
                        name: "echo".to_string(),
                        args: vec![BashNode::StringLit("matched".to_string())],
                    }],
                    else_body: vec![],
                };
                if node.depth() <= self.max_depth {
                    programs.push(self.node_to_generated(&node));
                }
            }
        }

        // Generate for loops with file patterns
        if self.max_depth >= 2 {
            let node = BashNode::For {
                var: "f".to_string(),
                items: vec![BashNode::StringLit("*.txt".to_string())],
                body: vec![BashNode::Command {
                    name: "echo".to_string(),
                    args: vec![BashNode::Variable("f".to_string())],
                }],
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        // Generate nested pipes (depth 3+)
        if self.max_depth >= 3 {
            let node = BashNode::Pipe {
                left: Box::new(BashNode::Command {
                    name: "cat".to_string(),
                    args: vec![BashNode::StringLit("/etc/passwd".to_string())],
                }),
                right: Box::new(BashNode::Pipe {
                    left: Box::new(BashNode::Command {
                        name: "grep".to_string(),
                        args: vec![BashNode::StringLit("root".to_string())],
                    }),
                    right: Box::new(BashNode::Command {
                        name: "wc".to_string(),
                        args: vec![BashNode::StringLit("-l".to_string())],
                    }),
                }),
            };
            if node.depth() <= self.max_depth {
                programs.push(self.node_to_generated(&node));
            }
        }

        programs
    }

    fn node_to_generated(&self, node: &BashNode) -> GeneratedCode {
        GeneratedCode {
            code: node.to_code(),
            language: Language::Bash,
            ast_depth: node.depth(),
            features: node.features(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_node_assignment() {
        let node = BashNode::Assignment {
            name: "x".to_string(),
            value: Box::new(BashNode::IntLit(42)),
        };
        assert_eq!(node.to_code(), "x=42");
    }

    #[test]
    fn test_bash_node_string_assignment() {
        let node = BashNode::Assignment {
            name: "msg".to_string(),
            value: Box::new(BashNode::StringLit("hello world".to_string())),
        };
        assert_eq!(node.to_code(), "msg=\"hello world\"");
    }

    #[test]
    fn test_bash_node_echo() {
        let node = BashNode::Command {
            name: "echo".to_string(),
            args: vec![BashNode::StringLit("hello".to_string())],
        };
        assert_eq!(node.to_code(), "echo hello");
    }

    #[test]
    fn test_bash_node_if() {
        let node = BashNode::If {
            condition: Box::new(BashNode::Test {
                double: false,
                expr: Box::new(BashNode::Compare {
                    left: Box::new(BashNode::Variable("x".to_string())),
                    op: BashCompareOp::NumEq,
                    right: Box::new(BashNode::IntLit(1)),
                }),
            }),
            then_body: vec![BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::StringLit("yes".to_string())],
            }],
            else_body: vec![],
        };
        let code = node.to_code();
        assert!(code.contains("if [ $x -eq 1 ]"));
        assert!(code.contains("then"));
        assert!(code.contains("fi"));
    }

    #[test]
    fn test_bash_node_for() {
        let node = BashNode::For {
            var: "i".to_string(),
            items: vec![BashNode::IntLit(1), BashNode::IntLit(2)],
            body: vec![BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::Variable("i".to_string())],
            }],
        };
        let code = node.to_code();
        assert!(code.contains("for i in 1 2"));
        assert!(code.contains("do"));
        assert!(code.contains("done"));
    }

    #[test]
    fn test_bash_node_while() {
        let node = BashNode::While {
            condition: Box::new(BashNode::Test {
                double: true,
                expr: Box::new(BashNode::Compare {
                    left: Box::new(BashNode::Variable("x".to_string())),
                    op: BashCompareOp::NumGt,
                    right: Box::new(BashNode::IntLit(0)),
                }),
            }),
            body: vec![BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::Variable("x".to_string())],
            }],
        };
        let code = node.to_code();
        assert!(code.contains("while [[ $x -gt 0 ]]"));
        assert!(code.contains("done"));
    }

    #[test]
    fn test_bash_node_function() {
        let node = BashNode::Function {
            name: "greet".to_string(),
            body: vec![BashNode::Command {
                name: "echo".to_string(),
                args: vec![BashNode::StringLit("Hello".to_string())],
            }],
        };
        let code = node.to_code();
        assert!(code.contains("greet()"));
        assert!(code.contains("{"));
        assert!(code.contains("}"));
    }

    #[test]
    fn test_bash_node_arithmetic() {
        let node = BashNode::Arithmetic(Box::new(BashNode::ArithOp {
            left: Box::new(BashNode::IntLit(1)),
            op: BashArithOp::Add,
            right: Box::new(BashNode::IntLit(2)),
        }));
        assert_eq!(node.to_code(), "$((1 + 2))");
    }

    #[test]
    fn test_bash_node_pipe() {
        let node = BashNode::Pipe {
            left: Box::new(BashNode::Command {
                name: "ls".to_string(),
                args: vec![],
            }),
            right: Box::new(BashNode::Command {
                name: "wc".to_string(),
                args: vec![BashNode::StringLit("-l".to_string())],
            }),
        };
        assert_eq!(node.to_code(), "ls | wc -l");
    }

    #[test]
    fn test_bash_node_depth() {
        let simple = BashNode::IntLit(1);
        assert_eq!(simple.depth(), 1);

        let assign = BashNode::Assignment {
            name: "x".to_string(),
            value: Box::new(BashNode::IntLit(1)),
        };
        assert_eq!(assign.depth(), 2);
    }

    #[test]
    fn test_bash_compare_op_all() {
        let ops = BashCompareOp::all();
        assert!(ops.len() >= 6);
    }

    #[test]
    fn test_bash_arith_op_all() {
        let ops = BashArithOp::all();
        assert_eq!(ops.len(), 5);
    }

    #[test]
    fn test_bash_enumerator_generates_programs() {
        let enumerator = BashEnumerator::new(3);
        let programs = enumerator.enumerate_programs();
        assert!(!programs.is_empty());

        for prog in &programs {
            assert_eq!(prog.language, Language::Bash);
            assert!(prog.ast_depth <= 3);
        }
    }

    #[test]
    fn test_bash_enumerator_depth_1() {
        let enumerator = BashEnumerator::new(1);
        let programs = enumerator.enumerate_programs();

        for prog in &programs {
            assert!(prog.ast_depth <= 1);
        }
    }

    #[test]
    fn test_bash_enumerator_depth_2() {
        let enumerator = BashEnumerator::new(2);
        let programs = enumerator.enumerate_programs();

        // Should have more programs than depth 1
        let depth1_count = BashEnumerator::new(1).enumerate_programs().len();
        assert!(programs.len() > depth1_count);
    }

    #[test]
    fn test_bash_node_features() {
        let node = BashNode::If {
            condition: Box::new(BashNode::Test {
                double: false,
                expr: Box::new(BashNode::Variable("x".to_string())),
            }),
            then_body: vec![],
            else_body: vec![],
        };
        let features = node.features();
        assert!(features.contains(&"if".to_string()));
    }
}
