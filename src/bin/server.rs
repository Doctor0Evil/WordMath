use axum::{
    extract::Query,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tower::ServiceBuilder;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use word_math_guard::{analyze_message, WordMathConfig};

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
}

#[tokio::main]
async fn main() {
    // Initialize logging.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    // Create router with a single /analyze endpoint.
    let app = Router::new()
        .route("/analyze", get(analyze_handler))
        .layer(ServiceBuilder::new());

    // Bind to localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    info!("listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn analyze_handler(Query(params): Query<AnalyzeParams>) -> Json<AnalyzeResponse> {
    // Default configuration; could be made configurable via env or config files.
    let cfg = WordMathConfig::default();

    let analysis = analyze_message(&params.message, &params.topic, cfg);

    Json(AnalyzeResponse {
        y_repetition: analysis.y_repetition,
        z_drift: analysis.z_drift,
        score: analysis.score,
    })
}
