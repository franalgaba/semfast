use crate::Result;
use crate::artifact::Index;
use crate::query::QueryTimings;
use crate::query::{Query, QueryMode};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkQuery {
    pub text: String,
    pub expected_chunk_id: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchReport {
    pub query_count: usize,
    pub harness: HarnessReport,
    pub search_only: LatencyReport,
    pub end_to_end: LatencyReport,
    pub components: ComponentLatencyReport,
    pub quality: QualityReport,
    pub artifact: ArtifactReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HarnessReport {
    pub embedding_model: String,
    pub vector_backend: String,
    pub warmed: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LatencyReport {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ComponentLatencyReport {
    pub embedding_ms: LatencyReport,
    pub vector_search_ms: LatencyReport,
    pub bm25_ms: LatencyReport,
    pub filtering_ms: LatencyReport,
    pub fusion_ms: LatencyReport,
    pub hydration_ms: LatencyReport,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct QualityReport {
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    pub mrr: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ArtifactReport {
    pub chunk_count: usize,
    pub artifact_size_bytes: u64,
    pub load_time_ms: f64,
}

pub struct BenchmarkRunner<'a> {
    index: &'a Index,
    artifact_size_bytes: u64,
    load_time_ms: f64,
    top_k: usize,
    alpha: f32,
    mode: QueryMode,
}

impl<'a> BenchmarkRunner<'a> {
    pub fn new(index: &'a Index) -> Self {
        Self {
            index,
            artifact_size_bytes: 0,
            load_time_ms: 0.0,
            top_k: 5,
            alpha: 0.7,
            mode: QueryMode::Hybrid,
        }
    }

    pub fn with_artifact_stats(
        index: &'a Index,
        artifact_path: &Path,
        load_time_ms: f64,
    ) -> Result<Self> {
        Ok(Self {
            index,
            artifact_size_bytes: directory_size(artifact_path)?,
            load_time_ms,
            top_k: 5,
            alpha: 0.7,
            mode: QueryMode::Hybrid,
        })
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn with_mode(mut self, mode: QueryMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn run(&self, queries: &[BenchmarkQuery]) -> Result<BenchReport> {
        self.warm_up(queries)?;

        let query_embeddings = queries
            .iter()
            .map(|query| self.index.embed_query(&query.text))
            .collect::<Result<Vec<_>>>()?;
        let mut search_only_ms = Vec::with_capacity(queries.len());
        for embedding in &query_embeddings {
            let started_at = Instant::now();
            let _ = self.index.search_vector(embedding, self.top_k)?;
            search_only_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
        }

        let mut end_to_end_ms = Vec::with_capacity(queries.len());
        let mut component_samples = ComponentSamples::with_capacity(queries.len());
        let mut recall_at_3_hits = 0;
        let mut recall_at_5_hits = 0;
        let mut reciprocal_rank_sum = 0.0;
        let mut expected_count = 0;

        for benchmark_query in queries {
            let query = Query {
                text: benchmark_query.text.clone(),
                top_k: self.top_k,
                alpha: self.alpha,
                filter: None,
                mode: self.mode,
            };

            let started_at = Instant::now();
            let measured = self.index.query_measured(query)?;
            end_to_end_ms.push(started_at.elapsed().as_secs_f64() * 1_000.0);
            component_samples.push(&measured.timings);

            if let Some(expected_chunk_id) = benchmark_query.expected_chunk_id {
                expected_count += 1;
                if measured
                    .results
                    .iter()
                    .take(3)
                    .any(|result| result.id.0 == expected_chunk_id)
                {
                    recall_at_3_hits += 1;
                }
                if measured
                    .results
                    .iter()
                    .take(5)
                    .any(|result| result.id.0 == expected_chunk_id)
                {
                    recall_at_5_hits += 1;
                }

                if let Some(rank) = measured
                    .results
                    .iter()
                    .position(|result| result.id.0 == expected_chunk_id)
                {
                    reciprocal_rank_sum += 1.0 / (rank + 1) as f64;
                }
            }
        }

        Ok(BenchReport {
            query_count: queries.len(),
            harness: HarnessReport {
                embedding_model: self.index.manifest().embedding_model.clone(),
                vector_backend: self.index.manifest().vector_backend.clone(),
                warmed: true,
            },
            search_only: LatencyReport::from_samples(&mut search_only_ms),
            end_to_end: LatencyReport::from_samples(&mut end_to_end_ms),
            components: component_samples.into_report(),
            quality: QualityReport {
                recall_at_3: ratio(recall_at_3_hits, expected_count),
                recall_at_5: ratio(recall_at_5_hits, expected_count),
                mrr: if expected_count == 0 {
                    0.0
                } else {
                    reciprocal_rank_sum / expected_count as f64
                },
            },
            artifact: ArtifactReport {
                chunk_count: self.index.manifest().chunk_count,
                artifact_size_bytes: self.artifact_size_bytes,
                load_time_ms: self.load_time_ms,
            },
        })
    }

    fn warm_up(&self, queries: &[BenchmarkQuery]) -> Result<()> {
        let warmup_text = queries
            .first()
            .map(|query| query.text.as_str())
            .unwrap_or("semfast warmup query");
        let warmup_embedding = self.index.embed_query(warmup_text)?;
        let _ = self.index.search_vector(&warmup_embedding, 5)?;
        let _ = self.index.query_measured(Query {
            text: warmup_text.to_string(),
            top_k: self.top_k,
            alpha: self.alpha,
            filter: None,
            mode: self.mode,
        })?;
        Ok(())
    }
}

impl LatencyReport {
    fn from_samples(samples: &mut [f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        samples.sort_by(f64::total_cmp);

        Self {
            p50_ms: percentile(samples, 0.50),
            p95_ms: percentile(samples, 0.95),
            p99_ms: percentile(samples, 0.99),
        }
    }
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

struct ComponentSamples {
    embedding_ms: Vec<f64>,
    vector_search_ms: Vec<f64>,
    bm25_ms: Vec<f64>,
    filtering_ms: Vec<f64>,
    fusion_ms: Vec<f64>,
    hydration_ms: Vec<f64>,
}

impl ComponentSamples {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            embedding_ms: Vec::with_capacity(capacity),
            vector_search_ms: Vec::with_capacity(capacity),
            bm25_ms: Vec::with_capacity(capacity),
            filtering_ms: Vec::with_capacity(capacity),
            fusion_ms: Vec::with_capacity(capacity),
            hydration_ms: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, timings: &QueryTimings) {
        self.embedding_ms.push(timings.embedding_ms);
        self.vector_search_ms.push(timings.vector_search_ms);
        self.bm25_ms.push(timings.bm25_ms);
        self.filtering_ms.push(timings.filtering_ms);
        self.fusion_ms.push(timings.fusion_ms);
        self.hydration_ms.push(timings.hydration_ms);
    }

    fn into_report(mut self) -> ComponentLatencyReport {
        ComponentLatencyReport {
            embedding_ms: LatencyReport::from_samples(&mut self.embedding_ms),
            vector_search_ms: LatencyReport::from_samples(&mut self.vector_search_ms),
            bm25_ms: LatencyReport::from_samples(&mut self.bm25_ms),
            filtering_ms: LatencyReport::from_samples(&mut self.filtering_ms),
            fusion_ms: LatencyReport::from_samples(&mut self.fusion_ms),
            hydration_ms: LatencyReport::from_samples(&mut self.hydration_ms),
        }
    }
}

fn directory_size(path: &Path) -> Result<u64> {
    let mut total_size = 0;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total_size += metadata.len();
        } else if metadata.is_dir() {
            total_size += directory_size(&entry.path())?;
        }
    }
    Ok(total_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;
    use crate::{IndexBuilder, IndexDocument};

    #[test]
    fn benchmark_report_includes_required_sections() {
        let temp_dir = tempfile::tempdir().unwrap();
        let documents = vec![IndexDocument::new(
            "doc",
            "Refund policy for damaged shipment claims.",
            Metadata::empty(),
        )];
        IndexBuilder::new()
            .build_to_path(&documents, temp_dir.path())
            .unwrap();
        let index = Index::load(temp_dir.path()).unwrap();
        let queries = [BenchmarkQuery {
            text: "damaged shipment refund".to_string(),
            expected_chunk_id: Some(1),
        }];

        let report = BenchmarkRunner::with_artifact_stats(&index, temp_dir.path(), 1.0)
            .unwrap()
            .run(&queries)
            .unwrap();

        assert_eq!(report.query_count, 1);
        assert_eq!(report.artifact.chunk_count, 1);
        assert!(report.artifact.artifact_size_bytes > 0);
        assert!(report.quality.recall_at_5 > 0.0);
    }
}
