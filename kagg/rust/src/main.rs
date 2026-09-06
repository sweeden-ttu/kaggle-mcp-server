mod database;
mod expert_registry;
mod kmap;
mod loop_controller;

use clap::{Parser, Subcommand};
use std::collections::HashMap;

#[derive(Parser)]
#[command(name = "kagg", version, about = "MLSysEng MoE CLI – Rust edition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List registered experts
    ListExperts {
        #[arg(long)]
        db: Option<String>,
    },
    /// Show system statistics
    Stats {
        #[arg(long)]
        db: Option<String>,
    },
    /// Build and simplify a K-map
    Kmap {
        /// Number of variables (2–8)
        #[arg(short, long, default_value_t = 4)]
        vars: usize,
        /// Comma-separated minterm indices
        #[arg(short, long)]
        minterms: String,
        /// Comma-separated don't-care indices
        #[arg(short, long)]
        dont_cares: Option<String>,
    },
    /// Check logical equivalence of two expressions via predicate table
    Equiv {
        /// First expression, e.g. "And(A, B)"
        #[arg(long)]
        a: String,
        /// Second expression, e.g. "Or(Not(A), Not(B))"
        #[arg(long)]
        b: String,
    },
    /// Run the convergence loop for a competition
    Evolve {
        /// Competition slug
        competition: String,
        #[arg(long, default_value_t = 10)]
        max_iter: usize,
        #[arg(long, default_value_t = 0.001)]
        epsilon: f64,
        #[arg(long)]
        db: Option<String>,
    },
}

fn parse_minterms(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

/// Minimal recursive-descent parser for BoolExpr from a string like
/// "And(A, Not(B))", "Or(Implies(A,B), C)", "Iff(A, B)", "Xor(A,B)".
fn parse_bool_expr(input: &str) -> Result<kmap::BoolExpr, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("empty expression".into());
    }

    if input == "true" || input == "True" || input == "1" {
        return Ok(kmap::BoolExpr::Const(true));
    }
    if input == "false" || input == "False" || input == "0" {
        return Ok(kmap::BoolExpr::Const(false));
    }

    // Function-call form: Name(args...)
    if let Some(paren) = input.find('(') {
        let name = input[..paren].trim();
        if !input.ends_with(')') {
            return Err(format!("unmatched parenthesis in: {}", input));
        }
        let inner = &input[paren + 1..input.len() - 1];

        match name {
            "Not" => {
                let arg = parse_bool_expr(inner)?;
                return Ok(kmap::BoolExpr::Not(Box::new(arg)));
            }
            "And" | "Or" | "Implies" | "Iff" | "Xor" => {
                let (left, right) = split_top_level_comma(inner)?;
                let a = parse_bool_expr(&left)?;
                let b = parse_bool_expr(&right)?;
                return match name {
                    "And" => Ok(kmap::BoolExpr::And(Box::new(a), Box::new(b))),
                    "Or" => Ok(kmap::BoolExpr::Or(Box::new(a), Box::new(b))),
                    "Implies" => Ok(kmap::BoolExpr::Implies(Box::new(a), Box::new(b))),
                    "Iff" => Ok(kmap::BoolExpr::Iff(Box::new(a), Box::new(b))),
                    "Xor" => Ok(kmap::BoolExpr::Xor(Box::new(a), Box::new(b))),
                    _ => unreachable!(),
                };
            }
            _ => return Err(format!("unknown function: {}", name)),
        }
    }

    // Bare variable name
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Ok(kmap::BoolExpr::Var(input.to_string()));
    }

    Err(format!("cannot parse: {}", input))
}

fn split_top_level_comma(s: &str) -> Result<(String, String), String> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                return Ok((s[..i].to_string(), s[i + 1..].to_string()));
            }
            _ => {}
        }
    }
    Err(format!("no top-level comma found in: {}", s))
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::ListExperts { db } => {
            let database = database::Database::open(db.as_deref())
                .expect("failed to open database");
            let experts = database.get_all_experts().expect("failed to list experts");
            if experts.is_empty() {
                println!("No experts registered. Run extract-knowledge first.");
            } else {
                for e in &experts {
                    println!("• {} ({})", e.expert_name, e.slug);
                    println!("  strategy: {}", e.strategy);
                    println!("  skills: {}", e.skills.join(", "));
                    println!();
                }
            }
        }

        Commands::Stats { db } => {
            let database = database::Database::open(db.as_deref())
                .expect("failed to open database");
            let stats = database.get_stats().expect("failed to get stats");
            println!("{}", serde_json::to_string_pretty(&stats).unwrap());
        }

        Commands::Kmap {
            vars,
            minterms,
            dont_cares,
        } => {
            let mt = parse_minterms(&minterms);
            let dc = dont_cares.map(|s| parse_minterms(&s));
            let km = kmap::KMap::new(&mt, vars, None, dc);
            println!("{}", km.display());
            let simplified = kmap::simplify(&km);
            println!("Simplified: {}", simplified.sop);
            println!(
                "Prime implicants: {}",
                simplified
                    .prime_implicants
                    .iter()
                    .map(|pi| pi.expression.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "Essential PIs: {}",
                simplified
                    .essential_pis
                    .iter()
                    .map(|pi| pi.expression.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        Commands::Equiv { a, b } => {
            let expr_a = parse_bool_expr(&a).expect("failed to parse expression A");
            let expr_b = parse_bool_expr(&b).expect("failed to parse expression B");
            let result = kmap::equivalence_table(&expr_a, &expr_b);
            print!("{}", result);
        }

        Commands::Evolve {
            competition,
            max_iter,
            epsilon,
            db,
        } => {
            let database = database::Database::open(db.as_deref())
                .expect("failed to open database");

            let mut rng_state: u64 = 42;
            let step_fn = |iteration: usize, prev: Option<&[f64]>| -> (Vec<f64>, serde_json::Value) {
                let mut vec = vec![0.0f64; 5];
                for v in &mut vec {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *v = (rng_state >> 33) as f64 / (u32::MAX as f64);
                }
                if let Some(p) = prev {
                    let decay = 0.5f64.powi(iteration as i32 + 1);
                    for (i, v) in vec.iter_mut().enumerate() {
                        if i < p.len() {
                            *v = p[i] + (0.5 - p[i]) * (1.0 - decay);
                        }
                    }
                }
                let meta = serde_json::json!({"iteration": iteration});
                (vec, meta)
            };

            let result = loop_controller::run_loop(
                &database,
                &competition,
                step_fn,
                epsilon,
                max_iter,
                3,
            );
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
    }
}
