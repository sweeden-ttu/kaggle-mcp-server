//! Aho-Corasick multi-pattern automaton for matching conference names,
//! venue keywords, and top ML Systems Engineering datasets.
//!
//! The automaton is constructed once and scans any input text in O(n + m)
//! time (n = text length, m = total matches), regardless of how many
//! patterns are loaded.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Categories of patterns the automaton recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    Conference,
    Dataset,
}

impl std::fmt::Display for PatternCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternCategory::Conference => write!(f, "conference"),
            PatternCategory::Dataset => write!(f, "dataset"),
        }
    }
}

/// A single match found by the automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcMatch {
    pub pattern: String,
    pub category: PatternCategory,
    pub start: usize,
    pub end: usize,
}

/// Summary of a scan over a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub matches: Vec<AcMatch>,
    pub category_counts: BTreeMap<String, usize>,
    pub unique_patterns: Vec<String>,
}

// -----------------------------------------------------------------------
// Pattern definitions
// -----------------------------------------------------------------------

fn conference_patterns() -> Vec<&'static str> {
    vec![
        "AAAI",
        "AAAI Conference",
        "AAAI 2027",
        "International Conference on Learning Representations",
        "International Learning Representations",
        "ICLR",
        "ICLR 2027",
        "NeurIPS",
        "Neural Information Processing Systems",
        "ICML",
        "International Conference on Machine Learning",
        "CVPR",
        "Computer Vision and Pattern Recognition",
        "ACL",
        "Association for Computational Linguistics",
        "EMNLP",
        "Empirical Methods in Natural Language Processing",
        "KDD",
        "Knowledge Discovery and Data Mining",
        "SIGMOD",
        "VLDB",
        "IJCAI",
        "International Joint Conference on Artificial Intelligence",
        "MLSys",
        "Machine Learning and Systems",
    ]
}

/// Top-100 ML Systems Engineering datasets — the canonical benchmarks
/// an ML systems engineer encounters across competitions, papers, and
/// production workloads.
fn dataset_patterns() -> Vec<&'static str> {
    vec![
        // Tabular / structured
        "Titanic",
        "House Prices",
        "Spaceship Titanic",
        "Store Sales",
        "Playground Series",
        "Porto Seguro",
        "IEEE-CIS Fraud Detection",
        "Santander Customer",
        "Home Credit Default Risk",
        "Elo Merchant",
        "Microsoft Malware Prediction",
        "Corporacion Favorita",
        "Instacart Market Basket",
        "Walmart Sales",
        "Rossmann Store Sales",
        "Allstate Claims",
        "Prudential Life Insurance",
        "Bosch Production Line",
        "Talking Data",
        "Avazu CTR",
        "Criteo Display Ads",
        // NLP
        "GLUE",
        "SuperGLUE",
        "SQuAD",
        "CoNLL",
        "SNLI",
        "MultiNLI",
        "SST-2",
        "MNLI",
        "Quora Question Pairs",
        "Natural Questions",
        "TriviaQA",
        "CommonCrawl",
        "C4 Dataset",
        "The Pile",
        "RedPajama",
        "MMLU",
        "HellaSwag",
        "WinoGrande",
        "ARC Challenge",
        "TruthfulQA",
        "HumanEval",
        "MBPP",
        "Jigsaw Toxic Comment",
        "Disaster Tweets",
        "Sentiment140",
        // Vision
        "ImageNet",
        "CIFAR-10",
        "CIFAR-100",
        "MNIST",
        "Fashion-MNIST",
        "COCO",
        "MS COCO",
        "Pascal VOC",
        "Open Images",
        "LSUN",
        "CelebA",
        "LFW",
        "Places365",
        "ADE20K",
        "Cityscapes",
        "KITTI",
        "Dogs vs Cats",
        "Plant Pathology",
        "Chest X-Ray",
        "Digit Recognizer",
        "Cassava Leaf Disease",
        // Audio / speech
        "LibriSpeech",
        "Common Voice",
        "AudioSet",
        "VoxCeleb",
        "RAVDESS",
        // Recommendation / ranking
        "MovieLens",
        "Netflix Prize",
        "Amazon Product Reviews",
        "Yelp Dataset",
        "Book-Crossing",
        "Steam Reviews",
        // Time series / forecasting
        "M5 Forecasting",
        "Web Traffic Forecasting",
        "COVID-19 Dataset",
        "Electricity Load",
        "ETTh1",
        "ETTm1",
        // Reinforcement learning
        "Atari 2600",
        "MuJoCo",
        "OpenAI Gym",
        "Procgen",
        "MineRL",
        // Systems / MLOps
        "MLPerf",
        "DAWNBench",
        "TPC-H",
        "TPC-DS",
        "BigBench",
        "DeepMind Lab",
        // Graph / relational
        "OGB",
        "Open Graph Benchmark",
        "Cora",
        "Citeseer",
        "PPI",
        "Reddit Dataset",
    ]
}

// -----------------------------------------------------------------------
// Automaton construction and scanning
// -----------------------------------------------------------------------

pub struct AcMatcher {
    automaton: AhoCorasick,
    patterns: Vec<(String, PatternCategory)>,
}

impl AcMatcher {
    /// Build the Aho-Corasick automaton from all conference + dataset patterns.
    /// Uses case-insensitive matching so "aaai", "AAAI", and "Aaai" all match.
    pub fn new() -> Self {
        let conf = conference_patterns();
        let ds = dataset_patterns();

        let mut patterns: Vec<(String, PatternCategory)> = Vec::new();
        for p in &conf {
            patterns.push((p.to_string(), PatternCategory::Conference));
        }
        for p in &ds {
            patterns.push((p.to_string(), PatternCategory::Dataset));
        }

        let pat_strings: Vec<&str> = patterns.iter().map(|(s, _)| s.as_str()).collect();

        let automaton = AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .match_kind(MatchKind::LeftmostLongest)
            .build(&pat_strings)
            .expect("failed to build Aho-Corasick automaton");

        Self {
            automaton,
            patterns,
        }
    }

    /// Scan input text and return all matches with positions and categories.
    pub fn scan(&self, text: &str) -> ScanResult {
        let mut matches = Vec::new();
        let mut category_counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut seen = std::collections::BTreeSet::new();

        for mat in self.automaton.find_iter(text) {
            let (ref pat, cat) = self.patterns[mat.pattern().as_usize()];
            matches.push(AcMatch {
                pattern: pat.clone(),
                category: cat,
                start: mat.start(),
                end: mat.end(),
            });
            *category_counts.entry(cat.to_string()).or_insert(0) += 1;
            seen.insert(pat.clone());
        }

        ScanResult {
            matches,
            category_counts,
            unique_patterns: seen.into_iter().collect(),
        }
    }

    /// Return the total number of patterns loaded.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Return all patterns for a given category.
    pub fn patterns_for(&self, category: PatternCategory) -> Vec<&str> {
        self.patterns
            .iter()
            .filter(|(_, c)| *c == category)
            .map(|(s, _)| s.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher() -> AcMatcher {
        AcMatcher::new()
    }

    #[test]
    fn test_pattern_count() {
        let m = matcher();
        assert!(m.pattern_count() > 100);
    }

    #[test]
    fn test_aaai_match() {
        let m = matcher();
        let result = m.scan("Submitted to AAAI 2027 conference");
        assert!(result.matches.iter().any(|hit| hit.pattern == "AAAI 2027"));
        assert!(result.category_counts.contains_key("conference"));
    }

    #[test]
    fn test_iclr_full_name() {
        let m = matcher();
        let result = m.scan("Published at the International Conference on Learning Representations");
        assert!(result.matches.iter().any(|hit| hit.pattern == "International Conference on Learning Representations"));
    }

    #[test]
    fn test_iclr_abbreviation() {
        let m = matcher();
        let result = m.scan("See our ICLR paper");
        assert!(result.matches.iter().any(|hit| hit.pattern == "ICLR"));
    }

    #[test]
    fn test_dataset_match_imagenet() {
        let m = matcher();
        let result = m.scan("We benchmark on ImageNet and CIFAR-10");
        let ds: Vec<&str> = result
            .matches
            .iter()
            .filter(|h| h.category == PatternCategory::Dataset)
            .map(|h| h.pattern.as_str())
            .collect();
        assert!(ds.contains(&"ImageNet"));
        assert!(ds.contains(&"CIFAR-10"));
    }

    #[test]
    fn test_dataset_titanic() {
        let m = matcher();
        let result = m.scan("Kaggle Titanic competition is a classic");
        assert!(result.matches.iter().any(|hit| hit.pattern == "Titanic"));
    }

    #[test]
    fn test_case_insensitive() {
        let m = matcher();
        let result = m.scan("we evaluated on mnist and cifar-100");
        let pats: Vec<&str> = result.matches.iter().map(|h| h.pattern.as_str()).collect();
        assert!(pats.contains(&"MNIST"));
        assert!(pats.contains(&"CIFAR-100"));
    }

    #[test]
    fn test_mixed_conference_and_dataset() {
        let m = matcher();
        let text = "Our AAAI 2027 paper evaluates on SQuAD, MMLU, and ImageNet using MLPerf benchmarks";
        let result = m.scan(text);
        assert!(result.category_counts.get("conference").unwrap_or(&0) > &0);
        assert!(result.category_counts.get("dataset").unwrap_or(&0) > &0);
        assert!(result.unique_patterns.len() >= 4);
    }

    #[test]
    fn test_no_false_positives_on_plain_text() {
        let m = matcher();
        let result = m.scan("The quick brown fox jumps over the lazy dog");
        assert!(result.matches.is_empty());
    }

    #[test]
    fn test_international_learning_representations() {
        let m = matcher();
        let result = m.scan("International Learning Representations venue");
        assert!(result.matches.iter().any(|hit| hit.pattern == "International Learning Representations"));
    }
}
