use crate::database::Database;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub fn l2_norm(vec: &[f64]) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let diff: Vec<f64> = a[..len].iter().zip(&b[..len]).map(|(x, y)| x - y).collect();
    l2_norm(&diff)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationResult {
    pub iteration: usize,
    pub l2_norm: f64,
    pub below_epsilon: bool,
    pub consecutive_converging: usize,
    pub converged: bool,
    pub should_stop: bool,
}

pub struct ConvergenceState {
    pub competition: String,
    pub epsilon: f64,
    pub max_iterations: usize,
    pub patience: usize,
    pub history: Vec<(Vec<f64>, IterationResult)>,
    pub consecutive_converging: usize,
}

impl ConvergenceState {
    pub fn new(competition: &str, epsilon: f64, max_iterations: usize, patience: usize) -> Self {
        Self {
            competition: competition.into(),
            epsilon,
            max_iterations,
            patience,
            history: Vec::new(),
            consecutive_converging: 0,
        }
    }

    pub fn current_iteration(&self) -> usize {
        self.history.len()
    }

    pub fn last_state(&self) -> Option<&[f64]> {
        self.history.last().map(|(v, _)| v.as_slice())
    }

    pub fn converged(&self) -> bool {
        self.consecutive_converging >= self.patience
    }

    pub fn should_stop(&self) -> bool {
        self.current_iteration() >= self.max_iterations || self.converged()
    }

    pub fn update(&mut self, state_vector: Vec<f64>) -> IterationResult {
        let norm = match self.last_state() {
            Some(prev) => l2_distance(&state_vector, prev),
            None => 0.0,
        };

        let below = self.history.last().is_some() && norm < self.epsilon;
        if below {
            self.consecutive_converging += 1;
        } else if self.history.last().is_some() {
            self.consecutive_converging = 0;
        }

        let result = IterationResult {
            iteration: self.current_iteration(),
            l2_norm: norm,
            below_epsilon: below,
            consecutive_converging: self.consecutive_converging,
            converged: self.converged(),
            should_stop: self.current_iteration() + 1 >= self.max_iterations || self.converged(),
        };

        self.history.push((state_vector, result.clone()));
        result
    }
}

pub fn run_loop<F>(
    db: &Database,
    competition: &str,
    mut step_fn: F,
    epsilon: f64,
    max_iterations: usize,
    patience: usize,
) -> serde_json::Value
where
    F: FnMut(usize, Option<&[f64]>) -> (Vec<f64>, serde_json::Value),
{
    let mut state = ConvergenceState::new(competition, epsilon, max_iterations, patience);

    while !state.should_stop() {
        let prev = state.last_state().map(|s| s.to_vec());
        let (new_vec, metadata) = step_fn(state.current_iteration(), prev.as_deref());
        let result = state.update(new_vec.clone());

        let _ = db.save_convergence_state(
            competition,
            result.iteration as i64,
            &new_vec,
            result.l2_norm,
            result.converged,
            &metadata,
        );

        if result.converged {
            break;
        }
    }

    let history: Vec<serde_json::Value> = state
        .history
        .iter()
        .map(|(_, r)| {
            json!({
                "iteration": r.iteration,
                "l2_norm": r.l2_norm,
                "below_epsilon": r.below_epsilon,
            })
        })
        .collect();

    json!({
        "competition": competition,
        "total_iterations": state.current_iteration(),
        "converged": state.converged(),
        "final_l2_norm": state.history.last().map(|(_, r)| r.l2_norm),
        "reason": if state.converged() { "converged" } else { "max_iterations" },
        "history": history,
    })
}
