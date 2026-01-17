use std::collections::{HashMap, HashSet};
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
        // alpha + beta should be <= 1.0 for the linear form below.
        Self { alpha: 0.5, beta: 0.5 }
    }
}

impl WordMathConfig {
    /// Load config from environment variables:
    /// WORD_MATH_ALPHA, WORD_MATH_BETA.
    /// Falls back to Default if parsing fails or vars are missing.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(alpha_str) = std::env::var("WORD_MATH_ALPHA") {
            if let Ok(alpha) = alpha_str.parse::<f64>() {
                cfg.alpha = alpha;
            }
        }

        if let Ok(beta_str) = std::env::var("WORD_MATH_BETA") {
            if let Ok(beta) = beta_str.parse::<f64>() {
                cfg.beta = beta;
            }
        }

        // Optional: normalize if alpha + beta > 1.0
        let sum = cfg.alpha + cfg.beta;
        if sum > 1.0 && sum > 0.0 {
            cfg.alpha /= sum;
            cfg.beta /= sum;
        }

        cfg
    }
}

/// Result of analyzing a single message.
#[derive(Debug, Clone)]
pub struct WordMathAnalysis {
    pub y_repetition: f64,
    pub z_drift: f64,
    pub score: f64,
}

/// Hex-stamped trace metadata for auditing.
#[derive(Debug, Clone)]
pub struct WordMathTrace {
    pub hex_id: String,
    pub message_len: usize,
    pub topic_len: usize,
}

/// Compute repetition density y = max_w c(w) / n for a message.
pub fn compute_repetition_density(message: &str) -> f64 {
    let words: Vec<String> = message
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

/// Jaccard-based topic drift baseline.
///
/// In a future version, you can plug in an embedding-based
/// distance here and keep this as a baseline for ablation.
pub fn compute_topic_drift(message: &str, topic: &str) -> f64 {
    let msg_words: HashSet<String> = message
        .unicode_words()
        .map(|w| w.to_lowercase())
        .collect();
    let topic_words: HashSet<String> = topic
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

/// Generate a simple hex ID for tracing.
/// This is intentionally minimal and not cryptographically strong.
pub fn generate_hex_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    format!("{:016x}", nanos)
}

/// Analyze a message given a topic string, returning y, z, f(y, z)
/// and a hex-stamped trace record.
pub fn analyze_message_with_trace(
    message: &str,
    topic: &str,
    cfg: WordMathConfig,
) -> (WordMathAnalysis, WordMathTrace) {
    let y = compute_repetition_density(message);
    let z = compute_topic_drift(message);
    let score = score_linear(y, z, cfg);

    let analysis = WordMathAnalysis {
        y_repetition: y,
        z_drift: z,
        score,
    };

    let trace = WordMathTrace {
        hex_id: generate_hex_id(),
        message_len: message.chars().count(),
        topic_len: topic.chars().count(),
    };

    (analysis, trace)
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
