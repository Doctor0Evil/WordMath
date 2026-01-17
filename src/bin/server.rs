use axum::{
    extract::{Query, State},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tower::ServiceBuilder;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use word_math_guard::{analyze_message_with_trace, WordMathConfig};

#[derive(Debug, Deserialize)]
struct AnalyzeParams {
    /// The user message to score.
    message: String,
    /// A short topic summary for the session.
    topic: String,
}

#[derive(Debug, Serialize)]
struct AnalyzeResponse {
    y_repetition: f64,
    z_drift: f64,
    score: f64,
    hex_id: String,
}

#[derive(Clone)]
struct AppState {
    cfg: WordMathConfig,
}

#[tokio::main]
async fn main() {
    // Initialize logging with env-based filter, e.g. RUST_LOG=info
    let subscriber = FmtSubscriber::builder()
        .with_env_filter("info")
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    // Load configuration from environment variables.
    let cfg = WordMathConfig::from_env();
    info!("Word-Math config: alpha={}, beta={}", cfg.alpha, cfg.beta);

    let state = AppState { cfg };

    // Create router with a single /analyze endpoint.
    let app = Router::new()
        .route("/analyze", get(analyze_handler))
        .with_state(Arc::new(state))
        .layer(ServiceBuilder::new());

    // Bind to localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn analyze_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AnalyzeParams>,
) -> Json<AnalyzeResponse> {
    let (analysis, trace) =
        analyze_message_with_trace(&params.message, &params.topic, state.cfg);

    // Hex-stamped, auditable trace log.
    info!(
        "HEX[{}]: y={:.4}, z={:.4}, score={:.4}, msg_len={}, topic_len={}",
        trace.hex_id,
        analysis.y_repetition,
        analysis.z_drift,
        analysis.score,
        trace.message_len,
        trace.topic_len
    );

    Json(AnalyzeResponse {
        y_repetition: analysis.y_repetition,
        z_drift: analysis.z_drift,
        score: analysis.score,
        hex_id: trace.hex_id,
    })
}
