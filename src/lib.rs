use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;

/// Configuration for the Word-Math scoring function f(y, z).
#[derive(Debug, Clone, Copy)]
pub struct WordMathConfig {
    /// Weight for repetition / contamination y
    pub alpha: f64,
    /// Weight for topic drift z
    pub beta: f64,
}

impl Default for WordMathConfig {
    fn default() -> Self {
        // Example values: repetition and drift weighted equally.
        // alpha + beta must be <= 1.0 for the linear form below.
        Self { alpha: 0.5, beta: 0.5 }
    }
}

/// Result of analyzing a single message.
#[derive(Debug, Clone)]
pub struct WordMathAnalysis {
    pub y_repetition: f64,
    pub z_drift: f64,
    pub score: f64,
}

/// Compute repetition density y = max_w c(w) / n for a message.
pub fn compute_repetition_density(message: &str) -> f64 {
    // Split into words using Unicode word boundaries.
    let words: Vec<&str> = message
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();

    let n = words.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for w in words {
        *counts.entry(w).or_insert(0) += 1;
    }

    let max_count = counts.values().copied().max().unwrap_or(0);
    max_count as f64 / n as f64
}

/// Compute topic drift z as a normalized distance in [0, 1].
///
/// This simple implementation uses Jaccard distance on word sets as a
/// stand-in for a semantic distance; in a full system you would plug
/// in embedding-based cosine distance instead.[web:51]
pub fn compute_topic_drift(message: &str, topic: &str) -> f64 {
    let msg_words: std::collections::HashSet<String> = message
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();
    let topic_words: std::collections::HashSet<String> = topic
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();

    if msg_words.is_empty() && topic_words.is_empty() {
        return 0.0;
    }
    if msg_words.is_empty() || topic_words.is_empty() {
        return 1.0;
    }

    let intersection_size = msg_words.intersection(&topic_words).count() as f64;
    let union_size = msg_words.union(&topic_words).count() as f64;

    let jaccard_similarity = if union_size > 0.0 {
        intersection_size / union_size
    } else {
        0.0
    };

    // Drift is 1 - similarity, already in [0, 1]
    1.0 - jaccard_similarity
}

/// Linear Word-Math scoring function:
/// f_lin(y, z) = 1 - alpha * y - beta * z
///
/// Assumes 0 <= alpha, beta, and alpha + beta <= 1. Returns a value in [0, 1].
pub fn score_linear(y: f64, z: f64, cfg: WordMathConfig) -> f64 {
    let mut score = 1.0 - cfg.alpha * y - cfg.beta * z;
    if score < 0.0 {
        score = 0.0;
    }
    if score > 1.0 {
        score = 1.0;
    }
    score
}

/// Analyze a message given a topic string, returning y, z and f(y, z).
pub fn analyze_message(message: &str, topic: &str, cfg: WordMathConfig) -> WordMathAnalysis {
    let y = compute_repetition_density(message);
    let z = compute_topic_drift(message, topic);
    let score = score_linear(y, z, cfg);

    WordMathAnalysis {
        y_repetition: y,
        z_drift: z,
        score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_density_empty() {
        let y = compute_repetition_density("");
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_repetition_density_basic() {
        let y = compute_repetition_density("hello hello world");
        // "hello" appears twice out of three words -> 2/3
        assert!((y - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_topic_drift_identical() {
        let z = compute_topic_drift("rust axum web server", "rust axum web server");
        assert!((z - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_topic_drift_disjoint() {
        let z = compute_topic_drift("rust", "banana apple");
        assert!((z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_linear_bounds() {
        let cfg = WordMathConfig::default();
        let s1 = score_linear(0.0, 0.0, cfg);
        let s2 = score_linear(1.0, 1.0, cfg);
        assert!(s1 <= 1.0 && s1 >= 0.0);
        assert!(s2 <= 1.0 && s2 >= 0.0);
    }
}
