use crate::chunking::ChunkId;
use crate::vector::VectorHit;
use crate::{bm25::Bm25Hit, query::QueryMode};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct RankedHit {
    pub chunk_id: ChunkId,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub lexical_score: Option<f32>,
}

pub fn fuse_hits(
    vector_hits: &[VectorHit],
    lexical_hits: &[Bm25Hit],
    top_k: usize,
    alpha: f32,
    mode: QueryMode,
) -> Vec<RankedHit> {
    let alpha = alpha.clamp(0.0, 1.0);
    let normalized_vectors = normalize_vector_hits(vector_hits);
    let normalized_lexical = normalize_bm25_hits(lexical_hits);
    let mut by_chunk_id: HashMap<ChunkId, RankedHit> = HashMap::new();

    for hit in vector_hits {
        let normalized_score = normalized_vectors
            .get(&hit.chunk_id)
            .copied()
            .unwrap_or_default();
        let score = match mode {
            QueryMode::VectorOnly => normalized_score,
            QueryMode::Hybrid => alpha * normalized_score,
            QueryMode::Bm25Only => 0.0,
        };
        by_chunk_id.insert(
            hit.chunk_id,
            RankedHit {
                chunk_id: hit.chunk_id,
                score,
                vector_score: Some(hit.score),
                lexical_score: None,
            },
        );
    }

    for hit in lexical_hits {
        let normalized_score = normalized_lexical
            .get(&hit.chunk_id)
            .copied()
            .unwrap_or_default();
        let score = match mode {
            QueryMode::Bm25Only => normalized_score,
            QueryMode::Hybrid => (1.0 - alpha) * normalized_score,
            QueryMode::VectorOnly => 0.0,
        };
        by_chunk_id
            .entry(hit.chunk_id)
            .and_modify(|existing| {
                existing.score += score;
                existing.lexical_score = Some(hit.score);
            })
            .or_insert(RankedHit {
                chunk_id: hit.chunk_id,
                score,
                vector_score: None,
                lexical_score: Some(hit.score),
            });
    }

    let mut hits = by_chunk_id.into_values().collect::<Vec<_>>();
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    hits.truncate(top_k);
    hits
}

fn normalize_vector_hits(hits: &[VectorHit]) -> HashMap<ChunkId, f32> {
    normalize_scores(hits.iter().map(|hit| (hit.chunk_id, hit.score)))
}

fn normalize_bm25_hits(hits: &[Bm25Hit]) -> HashMap<ChunkId, f32> {
    normalize_scores(hits.iter().map(|hit| (hit.chunk_id, hit.score)))
}

fn normalize_scores(scores: impl Iterator<Item = (ChunkId, f32)>) -> HashMap<ChunkId, f32> {
    let scores = scores.collect::<Vec<_>>();
    let Some(max_score) = scores
        .iter()
        .map(|(_, score)| *score)
        .max_by(f32::total_cmp)
    else {
        return HashMap::new();
    };

    if max_score <= 0.0 {
        return scores
            .into_iter()
            .map(|(chunk_id, _)| (chunk_id, 0.0))
            .collect();
    }

    scores
        .into_iter()
        .map(|(chunk_id, score)| (chunk_id, score / max_score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_fusion_combines_vector_and_lexical_scores() {
        let vector_hits = [VectorHit {
            chunk_id: ChunkId(1),
            score: 0.5,
        }];
        let lexical_hits = [Bm25Hit {
            chunk_id: ChunkId(2),
            score: 2.0,
        }];

        let hits = fuse_hits(&vector_hits, &lexical_hits, 2, 0.5, QueryMode::Hybrid);

        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|hit| hit.chunk_id == ChunkId(1)));
        assert!(hits.iter().any(|hit| hit.chunk_id == ChunkId(2)));
    }
}
