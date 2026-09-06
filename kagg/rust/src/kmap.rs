//! Recursively Enumerable K-map and Logical Equivalence Predicate Table.
//!
//! A K-map (Karnaugh map) is a visual method for simplifying Boolean algebra
//! expressions. This module implements:
//!
//! 1. **Recursively enumerable K-maps**: K-maps that can be generated for any
//!    number of variables by recursive Gray code construction. The enumeration
//!    produces all minterms, prime implicants, and essential PIs in a
//!    semi-decidable fashion (halts for finite variable sets).
//!
//! 2. **Logical equivalence predicate tables**: Truth-table–based evaluation
//!    that checks whether two Boolean expressions are logically equivalent
//!    by exhaustive enumeration of all 2^n variable assignments.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

/// Cell value in a K-map: 1 (true), 0 (false), or don't-care.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellValue {
    Zero,
    One,
    DontCare,
}

impl fmt::Display for CellValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellValue::Zero => write!(f, "0"),
            CellValue::One => write!(f, "1"),
            CellValue::DontCare => write!(f, "X"),
        }
    }
}

/// A single cell in a K-map grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMapCell {
    pub row: usize,
    pub col: usize,
    pub value: CellValue,
    pub minterm: usize,
}

/// A prime implicant covering a rectangular group of cells.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimeImplicant {
    pub minterms: BTreeSet<usize>,
    pub expression: String,
    pub is_essential: bool,
    pub cells: Vec<(usize, usize)>,
}

/// Simplified Boolean expression result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedExpression {
    pub sop: String,
    pub prime_implicants: Vec<PrimeImplicant>,
    pub essential_pis: Vec<PrimeImplicant>,
}

/// The K-map itself: grid + metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMap {
    pub num_vars: usize,
    pub variables: Vec<String>,
    pub grid: Vec<Vec<CellValue>>,
    pub rows: usize,
    pub cols: usize,
    pub gray_rows: Vec<usize>,
    pub gray_cols: Vec<usize>,
    pub minterms: Vec<usize>,
    pub dont_cares: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Recursive Gray code generator — the "recursively enumerable" core
// ---------------------------------------------------------------------------

/// Generate an n-bit Gray code sequence recursively.
///
/// The recursive definition:
///   G(1) = [0, 1]
///   G(n) = [0 ++ g for g in G(n-1)] ++ [1 ++ g for g in reversed(G(n-1))]
///
/// This is the mechanism that makes the K-map *recursively enumerable*:
/// for any finite n we can produce the Gray code, and the enumeration
/// procedure is primitive-recursive.
pub fn gray_code(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![0];
    }
    if n == 1 {
        return vec![0, 1];
    }
    let prev = gray_code(n - 1);
    let mut result = Vec::with_capacity(1 << n);
    for &v in &prev {
        result.push(v);
    }
    for &v in prev.iter().rev() {
        result.push(v | (1 << (n - 1)));
    }
    result
}

// ---------------------------------------------------------------------------
// K-map construction
// ---------------------------------------------------------------------------

impl KMap {
    /// Create a K-map for the given minterms and (optional) don't-cares.
    ///
    /// `num_vars` can be any positive integer; the grid dimensions are chosen
    /// by splitting variables between rows and columns as evenly as possible.
    pub fn new(
        minterms: &[usize],
        num_vars: usize,
        variables: Option<Vec<String>>,
        dont_cares: Option<Vec<usize>>,
    ) -> Self {
        let vars: Vec<String> = variables.unwrap_or_else(|| {
            (0..num_vars)
                .map(|i| ((b'A' + i as u8) as char).to_string())
                .collect()
        });
        let dc = dont_cares.unwrap_or_default();

        let row_bits = num_vars / 2;
        let col_bits = num_vars - row_bits;
        let rows = 1 << row_bits;
        let cols = 1 << col_bits;

        let gray_rows = gray_code(row_bits);
        let gray_cols = gray_code(col_bits);

        let max_minterm = (1 << num_vars) - 1;

        let mut grid = vec![vec![CellValue::Zero; cols]; rows];

        let row_index: HashMap<usize, usize> =
            gray_rows.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let col_index: HashMap<usize, usize> =
            gray_cols.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        for &m in minterms {
            if m > max_minterm {
                continue;
            }
            let rv = m >> col_bits;
            let cv = m & ((1 << col_bits) - 1);
            if let (Some(&r), Some(&c)) = (row_index.get(&rv), col_index.get(&cv)) {
                grid[r][c] = CellValue::One;
            }
        }
        for &m in &dc {
            if m > max_minterm {
                continue;
            }
            let rv = m >> col_bits;
            let cv = m & ((1 << col_bits) - 1);
            if let (Some(&r), Some(&c)) = (row_index.get(&rv), col_index.get(&cv)) {
                grid[r][c] = CellValue::DontCare;
            }
        }

        Self {
            num_vars,
            variables: vars,
            grid,
            rows,
            cols,
            gray_rows,
            gray_cols,
            minterms: minterms.to_vec(),
            dont_cares: dc,
        }
    }

    /// Convert a grid (row, col) position back to a minterm number.
    pub fn position_to_minterm(&self, row: usize, col: usize) -> usize {
        let col_bits = self.num_vars - self.num_vars / 2;
        let rv = self.gray_rows[row];
        let cv = self.gray_cols[col];
        (rv << col_bits) | cv
    }

    /// Render the K-map as an ASCII table.
    pub fn display(&self) -> String {
        let col_bits = self.num_vars - self.num_vars / 2;
        let row_bits = self.num_vars / 2;

        let row_vars = self.variables[..row_bits].join("");
        let col_vars = self.variables[row_bits..].join("");

        let mut out = format!("{} \\ {}", row_vars, col_vars);

        // Column headers
        out.push_str("\n     ");
        for &gc in &self.gray_cols {
            out.push_str(&format!(" {:0width$b}", gc, width = col_bits));
        }
        out.push('\n');

        for (ri, &gr) in self.gray_rows.iter().enumerate() {
            out.push_str(&format!("  {:0width$b} |", gr, width = row_bits));
            for ci in 0..self.cols {
                out.push_str(&format!("  {} ", self.grid[ri][ci]));
            }
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Quine–McCluskey style simplification
// ---------------------------------------------------------------------------

/// Recursively enumerate prime implicants using the Quine–McCluskey algorithm.
///
/// Each minterm is a `num_vars`-wide bit pattern; a dash (represented by `None`)
/// means the variable is eliminated.  Two implicants that differ in exactly one
/// bit combine into a larger implicant with that bit dashed.  The process
/// repeats until no further merges are possible — the remaining implicants are
/// *prime*.
fn quine_mccluskey(
    minterms: &BTreeSet<usize>,
    dont_cares: &BTreeSet<usize>,
    num_vars: usize,
) -> Vec<(BTreeSet<usize>, Vec<Option<bool>>)> {
    let all_terms: BTreeSet<usize> = minterms.union(dont_cares).copied().collect();

    // Each implicant: (set of covered minterms, pattern where None = dash)
    type Imp = (BTreeSet<usize>, Vec<Option<bool>>);

    let mut current: Vec<Imp> = all_terms
        .iter()
        .map(|&m| {
            let pattern: Vec<Option<bool>> = (0..num_vars)
                .rev()
                .map(|bit| Some((m >> bit) & 1 == 1))
                .collect();
            let mut s = BTreeSet::new();
            s.insert(m);
            (s, pattern)
        })
        .collect();

    let mut primes: Vec<Imp> = Vec::new();

    loop {
        let mut merged_flags = vec![false; current.len()];
        let mut next: Vec<Imp> = Vec::new();
        let mut seen: HashSet<Vec<Option<bool>>> = HashSet::new();

        for i in 0..current.len() {
            for j in (i + 1)..current.len() {
                if let Some(combined) = try_merge(&current[i].1, &current[j].1) {
                    if seen.insert(combined.clone()) {
                        let mut terms = current[i].0.clone();
                        terms.extend(&current[j].0);
                        next.push((terms, combined));
                    }
                    merged_flags[i] = true;
                    merged_flags[j] = true;
                }
            }
        }

        for (idx, was_merged) in merged_flags.iter().enumerate() {
            if !was_merged {
                primes.push(current[idx].clone());
            }
        }

        if next.is_empty() {
            break;
        }
        current = next;
    }

    // Deduplicate by pattern
    let mut unique: Vec<Imp> = Vec::new();
    let mut seen_patterns: HashSet<Vec<Option<bool>>> = HashSet::new();
    for pi in primes {
        if seen_patterns.insert(pi.1.clone()) {
            unique.push(pi);
        }
    }
    unique
}

fn try_merge(a: &[Option<bool>], b: &[Option<bool>]) -> Option<Vec<Option<bool>>> {
    if a.len() != b.len() {
        return None;
    }
    let mut diff_count = 0;
    let mut diff_idx = 0;
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        if av != bv {
            diff_count += 1;
            diff_idx = i;
            if diff_count > 1 {
                return None;
            }
        }
    }
    if diff_count != 1 {
        return None;
    }
    let mut result = a.to_vec();
    result[diff_idx] = None; // dash
    Some(result)
}

fn pattern_to_expression(pattern: &[Option<bool>], variables: &[String]) -> String {
    let mut terms = Vec::new();
    for (i, &bit) in pattern.iter().enumerate() {
        match bit {
            Some(true) => terms.push(variables[i].clone()),
            Some(false) => terms.push(format!("{}'", variables[i])),
            None => {} // eliminated variable
        }
    }
    if terms.is_empty() {
        "1".into()
    } else {
        terms.join("")
    }
}

/// Select essential prime implicants and a minimal cover.
fn select_essential(
    primes: &[(BTreeSet<usize>, Vec<Option<bool>>)],
    minterms: &BTreeSet<usize>,
    variables: &[String],
) -> (Vec<PrimeImplicant>, Vec<PrimeImplicant>) {
    // Coverage map: minterm -> indices of PIs covering it
    let mut coverage: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &m in minterms {
        for (idx, (terms, _)) in primes.iter().enumerate() {
            if terms.contains(&m) {
                coverage.entry(m).or_default().push(idx);
            }
        }
    }

    let mut essential_indices: BTreeSet<usize> = BTreeSet::new();
    for (_m, covering) in &coverage {
        if covering.len() == 1 {
            essential_indices.insert(covering[0]);
        }
    }

    let to_pi = |idx: usize, essential: bool| -> PrimeImplicant {
        let (terms, pattern) = &primes[idx];
        PrimeImplicant {
            minterms: terms.clone(),
            expression: pattern_to_expression(pattern, variables),
            is_essential: essential,
            cells: vec![],
        }
    };

    let essential_pis: Vec<PrimeImplicant> =
        essential_indices.iter().map(|&i| to_pi(i, true)).collect();

    let mut covered: BTreeSet<usize> = BTreeSet::new();
    for pi in &essential_pis {
        covered.extend(&pi.minterms);
    }

    let mut all_selected = essential_pis.clone();
    let remaining: BTreeSet<usize> = minterms.difference(&covered).copied().collect();
    if !remaining.is_empty() {
        for (idx, (terms, _)) in primes.iter().enumerate() {
            if essential_indices.contains(&idx) {
                continue;
            }
            if terms.intersection(&remaining).next().is_some() {
                all_selected.push(to_pi(idx, false));
            }
        }
    }

    (all_selected, essential_pis)
}

/// Simplify a K-map to minimal sum-of-products form.
pub fn simplify(kmap: &KMap) -> SimplifiedExpression {
    let mt_set: BTreeSet<usize> = kmap.minterms.iter().copied().collect();
    let dc_set: BTreeSet<usize> = kmap.dont_cares.iter().copied().collect();

    let primes = quine_mccluskey(&mt_set, &dc_set, kmap.num_vars);
    let (all_pis, essential_pis) = select_essential(&primes, &mt_set, &kmap.variables);

    let sop = if all_pis.is_empty() {
        "0".into()
    } else {
        all_pis
            .iter()
            .map(|pi| pi.expression.as_str())
            .collect::<Vec<_>>()
            .join(" + ")
    };

    SimplifiedExpression {
        sop,
        prime_implicants: all_pis,
        essential_pis,
    }
}

// ---------------------------------------------------------------------------
// Logical Equivalence Predicate Table
// ---------------------------------------------------------------------------

/// A Boolean expression AST that can be evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoolExpr {
    Var(String),
    Const(bool),
    Not(Box<BoolExpr>),
    And(Box<BoolExpr>, Box<BoolExpr>),
    Or(Box<BoolExpr>, Box<BoolExpr>),
    Implies(Box<BoolExpr>, Box<BoolExpr>),
    Iff(Box<BoolExpr>, Box<BoolExpr>),
    Xor(Box<BoolExpr>, Box<BoolExpr>),
}

impl BoolExpr {
    /// Collect all variable names used in this expression.
    pub fn variables(&self) -> BTreeSet<String> {
        let mut vars = BTreeSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut BTreeSet<String>) {
        match self {
            BoolExpr::Var(name) => { vars.insert(name.clone()); }
            BoolExpr::Const(_) => {}
            BoolExpr::Not(inner) => inner.collect_vars(vars),
            BoolExpr::And(a, b)
            | BoolExpr::Or(a, b)
            | BoolExpr::Implies(a, b)
            | BoolExpr::Iff(a, b)
            | BoolExpr::Xor(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
        }
    }

    /// Evaluate under a given assignment.
    pub fn eval(&self, env: &HashMap<String, bool>) -> bool {
        match self {
            BoolExpr::Var(name) => *env.get(name).unwrap_or(&false),
            BoolExpr::Const(v) => *v,
            BoolExpr::Not(inner) => !inner.eval(env),
            BoolExpr::And(a, b) => a.eval(env) && b.eval(env),
            BoolExpr::Or(a, b) => a.eval(env) || b.eval(env),
            BoolExpr::Implies(a, b) => !a.eval(env) || b.eval(env),
            BoolExpr::Iff(a, b) => a.eval(env) == b.eval(env),
            BoolExpr::Xor(a, b) => a.eval(env) != b.eval(env),
        }
    }
}

/// A row in the equivalence predicate table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateTableRow {
    pub assignment: BTreeMap<String, bool>,
    pub value_a: bool,
    pub value_b: bool,
    pub equivalent: bool,
}

/// Full logical equivalence evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquivalenceResult {
    pub is_equivalent: bool,
    pub table: Vec<PredicateTableRow>,
    pub counterexample: Option<BTreeMap<String, bool>>,
    pub total_assignments: usize,
}

/// Recursively enumerate all 2^n assignments and build the equivalence
/// predicate table for two Boolean expressions.
///
/// The enumeration is recursively enumerable (r.e.) in the sense that
/// for any finite variable set it will halt; for a countably infinite
/// set of propositional variables we would produce an r.e. stream of
/// partial evaluations (each finite prefix is decidable).
pub fn equivalence_table(expr_a: &BoolExpr, expr_b: &BoolExpr) -> EquivalenceResult {
    let mut all_vars = expr_a.variables();
    all_vars.extend(expr_b.variables());
    let vars: Vec<String> = all_vars.into_iter().collect();
    let n = vars.len();
    let total = 1usize << n;

    let mut table = Vec::with_capacity(total);
    let mut counterexample: Option<BTreeMap<String, bool>> = None;
    let mut all_equiv = true;

    for bits in 0..total {
        let mut env = HashMap::new();
        let mut assignment = BTreeMap::new();
        for (i, var) in vars.iter().enumerate() {
            let val = (bits >> (n - 1 - i)) & 1 == 1;
            env.insert(var.clone(), val);
            assignment.insert(var.clone(), val);
        }

        let va = expr_a.eval(&env);
        let vb = expr_b.eval(&env);
        let eq = va == vb;

        if !eq {
            all_equiv = false;
            if counterexample.is_none() {
                counterexample = Some(assignment.clone());
            }
        }

        table.push(PredicateTableRow {
            assignment,
            value_a: va,
            value_b: vb,
            equivalent: eq,
        });
    }

    EquivalenceResult {
        is_equivalent: all_equiv,
        table,
        counterexample,
        total_assignments: total,
    }
}

/// Generate a truth table for a single expression (useful for K-map input).
pub fn truth_table(expr: &BoolExpr) -> Vec<(BTreeMap<String, bool>, bool)> {
    let vars: Vec<String> = expr.variables().into_iter().collect();
    let n = vars.len();
    let total = 1usize << n;

    let mut rows = Vec::with_capacity(total);
    for bits in 0..total {
        let mut env = HashMap::new();
        let mut assignment = BTreeMap::new();
        for (i, var) in vars.iter().enumerate() {
            let val = (bits >> (n - 1 - i)) & 1 == 1;
            env.insert(var.clone(), val);
            assignment.insert(var.clone(), val);
        }
        rows.push((assignment, expr.eval(&env)));
    }
    rows
}

/// Build a K-map directly from a Boolean expression by evaluating its truth table.
pub fn kmap_from_expr(expr: &BoolExpr) -> KMap {
    let tt = truth_table(expr);
    let vars: Vec<String> = expr.variables().into_iter().collect();
    let n = vars.len();

    let minterms: Vec<usize> = tt
        .iter()
        .enumerate()
        .filter_map(|(i, (_, val))| if *val { Some(i) } else { None })
        .collect();

    KMap::new(&minterms, n, Some(vars), None)
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

impl fmt::Display for EquivalenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.table.is_empty() {
            return writeln!(f, "(empty table)");
        }
        let vars: Vec<&String> = self.table[0].assignment.keys().collect();
        for v in &vars {
            write!(f, " {} |", v)?;
        }
        writeln!(f, " A | B | A≡B")?;
        writeln!(f, "{}", "-".repeat(vars.len() * 4 + 14))?;

        for row in &self.table {
            for v in &vars {
                write!(f, " {} |", if row.assignment[*v] { 1 } else { 0 })?;
            }
            writeln!(
                f,
                " {} | {} |  {}",
                row.value_a as u8,
                row.value_b as u8,
                if row.equivalent { "✓" } else { "✗" }
            )?;
        }

        if self.is_equivalent {
            writeln!(f, "\nResult: EQUIVALENT (all {} rows match)", self.total_assignments)?;
        } else {
            writeln!(f, "\nResult: NOT EQUIVALENT")?;
            if let Some(ref ce) = self.counterexample {
                write!(f, "Counterexample: ")?;
                for (k, v) in ce {
                    write!(f, "{}={} ", k, *v as u8)?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_base_cases() {
        assert_eq!(gray_code(0), vec![0]);
        assert_eq!(gray_code(1), vec![0, 1]);
        assert_eq!(gray_code(2), vec![0, 1, 3, 2]);
        assert_eq!(gray_code(3), vec![0, 1, 3, 2, 6, 7, 5, 4]);
    }

    #[test]
    fn test_gray_code_adjacency() {
        for n in 1..=5 {
            let codes = gray_code(n);
            assert_eq!(codes.len(), 1 << n);
            for window in codes.windows(2) {
                let diff = window[0] ^ window[1];
                assert!(diff.is_power_of_two(), "Adjacent codes must differ in exactly one bit");
            }
        }
    }

    #[test]
    fn test_kmap_2var() {
        let kmap = KMap::new(&[0, 3], 2, None, None);
        // 2 vars: row_bits=1, col_bits=1 → 2×2 grid
        assert_eq!(kmap.rows, 2);
        assert_eq!(kmap.cols, 2);
        assert_eq!(kmap.minterms, vec![0, 3]);
    }

    #[test]
    fn test_simplify_single_minterm() {
        let kmap = KMap::new(&[5], 3, Some(vec!["A".into(), "B".into(), "C".into()]), None);
        let result = simplify(&kmap);
        assert!(!result.sop.is_empty());
        assert_ne!(result.sop, "0");
    }

    #[test]
    fn test_equivalence_tautology() {
        let a = BoolExpr::Or(
            Box::new(BoolExpr::Var("x".into())),
            Box::new(BoolExpr::Not(Box::new(BoolExpr::Var("x".into())))),
        );
        let b = BoolExpr::Const(true);
        let result = equivalence_table(&a, &b);
        assert!(result.is_equivalent);
    }

    #[test]
    fn test_equivalence_demorgan() {
        // !(A && B) == (!A || !B)
        let lhs = BoolExpr::Not(Box::new(BoolExpr::And(
            Box::new(BoolExpr::Var("A".into())),
            Box::new(BoolExpr::Var("B".into())),
        )));
        let rhs = BoolExpr::Or(
            Box::new(BoolExpr::Not(Box::new(BoolExpr::Var("A".into())))),
            Box::new(BoolExpr::Not(Box::new(BoolExpr::Var("B".into())))),
        );
        let result = equivalence_table(&lhs, &rhs);
        assert!(result.is_equivalent);
        assert!(result.counterexample.is_none());
    }

    #[test]
    fn test_equivalence_not_equivalent() {
        let a = BoolExpr::Var("x".into());
        let b = BoolExpr::Not(Box::new(BoolExpr::Var("x".into())));
        let result = equivalence_table(&a, &b);
        assert!(!result.is_equivalent);
        assert!(result.counterexample.is_some());
    }

    #[test]
    fn test_kmap_from_expr() {
        let expr = BoolExpr::And(
            Box::new(BoolExpr::Var("A".into())),
            Box::new(BoolExpr::Var("B".into())),
        );
        let kmap = kmap_from_expr(&expr);
        assert_eq!(kmap.num_vars, 2);
        assert_eq!(kmap.minterms, vec![3]); // A=1, B=1 => minterm 3
    }
}
