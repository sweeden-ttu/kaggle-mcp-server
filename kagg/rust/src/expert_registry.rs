use crate::database::{Database, Expert};
use regex::Regex;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

const DEFAULT_STRATEGY: &str = "Baseline → EDA → Feature Engineering → Model Selection → Submit";

pub fn default_loop_config() -> serde_json::Value {
    json!({
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": 0.001,
        "max_iterations": 10,
        "patience": 3
    })
}

fn skill_mapping() -> HashMap<&'static str, Vec<&'static str>> {
    let mut m = HashMap::new();
    m.insert("gradient descent", vec!["kaggle-model-trainer", "kaggle-optimizer"]);
    m.insert("neural network", vec!["kaggle-deep-learning", "kaggle-model-trainer"]);
    m.insert("deep learning", vec!["kaggle-deep-learning", "kaggle-model-trainer"]);
    m.insert("feature engineering", vec!["kaggle-preprocessor", "kaggle-feature-engineer"]);
    m.insert("ensemble", vec!["kaggle-ensemble", "kaggle-model-trainer"]);
    m.insert("cross validation", vec!["kaggle-validator", "kaggle-preprocessor"]);
    m.insert("regularization", vec!["kaggle-model-trainer", "kaggle-optimizer"]);
    m.insert("optimization", vec!["kaggle-optimizer", "kaggle-model-trainer"]);
    m.insert("transformer", vec!["kaggle-deep-learning", "kaggle-nlp"]);
    m.insert("attention mechanism", vec!["kaggle-deep-learning", "kaggle-nlp"]);
    m.insert("clustering", vec!["kaggle-unsupervised", "kaggle-preprocessor"]);
    m.insert("pca", vec!["kaggle-preprocessor", "kaggle-feature-engineer"]);
    m.insert("bayesian", vec!["kaggle-model-trainer", "kaggle-optimizer"]);
    m.insert("reinforcement learning", vec!["kaggle-rl", "kaggle-model-trainer"]);
    m.insert("generative model", vec!["kaggle-deep-learning", "kaggle-generative"]);
    m.insert("distributed training", vec!["kaggle-scaling", "kaggle-model-trainer"]);
    m.insert("mixture of experts", vec!["kaggle-moe", "kaggle-model-trainer"]);
    m
}

fn formula_for_concepts(concepts: &[String]) -> serde_json::Value {
    let joined = concepts.join(" ").to_lowercase();
    if ["classification", "precision", "recall", "f1"]
        .iter()
        .any(|kw| joined.contains(kw))
    {
        return json!({
            "objective": "minimize_cross_entropy_loss",
            "function": "L = -Σ y_i * log(ŷ_i)",
            "metrics": ["accuracy", "f1_score", "precision", "recall"]
        });
    }
    if ["regression", "mse", "rmse"]
        .iter()
        .any(|kw| joined.contains(kw))
    {
        return json!({
            "objective": "minimize_mse",
            "function": "L = (1/n) * Σ (y_i - ŷ_i)²",
            "metrics": ["rmse", "mae", "r2_score"]
        });
    }
    if ["clustering", "unsupervised", "pca"]
        .iter()
        .any(|kw| joined.contains(kw))
    {
        return json!({
            "objective": "minimize_inertia",
            "function": "J = Σ ||x_i - μ_k||²",
            "metrics": ["silhouette_score", "inertia"]
        });
    }
    json!({
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"]
    })
}

fn default_skills_path() -> String {
    std::env::var("KAGGLE_SKILLS_PATH")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{}/skills", home)
        })
}

pub fn slugify(name: &str) -> String {
    let re = Regex::new(r"[^\w\s-]").unwrap();
    let lowered = name.to_lowercase();
    let s = re.replace_all(&lowered, "");
    let re2 = Regex::new(r"[\s-]+").unwrap();
    re2.replace_all(&s, "_").trim_matches('_').to_string()
}

fn infer_skills(concepts: &[String]) -> Vec<String> {
    let base = default_skills_path();
    let mapping = skill_mapping();
    let mut skill_set: HashSet<String> = HashSet::new();

    for concept in concepts {
        let cl = concept.to_lowercase();
        for (pattern, skill_names) in &mapping {
            if cl.contains(pattern) {
                for s in skill_names {
                    skill_set.insert(format!("{}/{}", base, s));
                }
            }
        }
    }

    if skill_set.is_empty() {
        skill_set.insert(format!("{}/kaggle-preprocessor", base));
        skill_set.insert(format!("{}/kaggle-model-trainer", base));
    }

    let mut v: Vec<String> = skill_set.into_iter().collect();
    v.sort();
    v
}

fn infer_capabilities(title: &str, concepts: &[String]) -> Vec<String> {
    let mut caps = vec![format!("Expert knowledge in: {}", title)];
    let has = |pat: &str| concepts.iter().any(|c| c.to_lowercase().contains(pat));

    if has("model") { caps.push("Build baseline models quickly".into()); }
    if has("optim") { caps.push("Systematic hyperparameter search".into()); }
    if has("feature") { caps.push("Advanced feature engineering".into()); }
    if has("ensemble") { caps.push("Ensemble model construction".into()); }
    if has("neural") || has("deep") { caps.push("Deep learning architecture design".into()); }
    if has("valid") || has("cross") { caps.push("Robust cross-validation strategies".into()); }

    if caps.len() < 3 {
        caps.push("Data preprocessing and cleaning".into());
        caps.push("Model evaluation and selection".into());
    }
    caps
}

pub struct ExpertRegistry<'a> {
    db: &'a Database,
}

impl<'a> ExpertRegistry<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    pub fn create_expert_from_chapter(
        &self,
        chapter_number: i64,
        title: &str,
        concepts: &[String],
        chapter_id: Option<i64>,
    ) -> rusqlite::Result<Expert> {
        let name = format!("{:02}_{}", chapter_number, title);
        let slug = slugify(&name);

        let expert = Expert {
            id: 0,
            expert_name: name,
            slug: slug.clone(),
            chapter_id,
            capabilities: infer_capabilities(title, concepts),
            skills: infer_skills(concepts),
            strategy: DEFAULT_STRATEGY.into(),
            formula: formula_for_concepts(concepts),
            loop_config: default_loop_config(),
            created_at: None,
            updated_at: None,
        };

        self.db.upsert_expert(&expert)?;
        Ok(expert)
    }

    pub fn register_all_from_chapters(&self) -> rusqlite::Result<Vec<Expert>> {
        let chapters = self.db.get_all_chapters()?;
        let mut experts = Vec::new();
        for ch in &chapters {
            let expert = self.create_expert_from_chapter(
                ch.chapter_number,
                &ch.title,
                &ch.concepts,
                Some(ch.id),
            )?;
            experts.push(expert);
        }
        Ok(experts)
    }

    pub fn list_experts(&self) -> rusqlite::Result<Vec<Expert>> {
        self.db.get_all_experts()
    }

    pub fn get_expert(&self, slug: &str) -> rusqlite::Result<Option<Expert>> {
        self.db.get_expert(slug)
    }

    pub fn export_expert_json(&self, slug: &str) -> rusqlite::Result<Option<String>> {
        match self.db.get_expert(slug)? {
            Some(e) => {
                let export = json!({
                    "expert_name": e.expert_name,
                    "slug": e.slug,
                    "capabilities": e.capabilities,
                    "skills": e.skills,
                    "strategy": e.strategy,
                    "formula": e.formula,
                    "loop_config": e.loop_config,
                });
                Ok(Some(serde_json::to_string_pretty(&export).unwrap()))
            }
            None => Ok(None),
        }
    }
}
