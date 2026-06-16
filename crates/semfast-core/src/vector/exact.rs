use crate::chunking::ChunkId;
use crate::vector::{VectorHit, VectorIndex};
use crate::{Result, SemfastError};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExactVectorIndex {
    dimensions: usize,
    vectors: Vec<StoredVector>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredVector {
    chunk_id: ChunkId,
    values: Vec<f32>,
}

impl ExactVectorIndex {
    pub fn build(vectors: &[f32], chunk_ids: &[ChunkId], dimensions: usize) -> Result<Self> {
        if vectors.len() != chunk_ids.len() * dimensions {
            return Err(SemfastError::InvalidInput(
                "vector buffer length does not match ids and dimensions".to_string(),
            ));
        }

        let vectors = chunk_ids
            .iter()
            .enumerate()
            .map(|(index, &chunk_id)| {
                let start = index * dimensions;
                let end = start + dimensions;
                StoredVector {
                    chunk_id,
                    values: vectors[start..end].to_vec(),
                }
            })
            .collect();

        Ok(Self {
            dimensions,
            vectors,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }
}

impl VectorIndex for ExactVectorIndex {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>> {
        if query.len() != self.dimensions {
            return Err(SemfastError::InvalidInput(format!(
                "query has {} dimensions, expected {}",
                query.len(),
                self.dimensions
            )));
        }

        let mut hits = self
            .vectors
            .iter()
            .map(|stored| VectorHit {
                chunk_id: stored.chunk_id,
                score: dot_product(query, &stored.values),
            })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(top_k);
        Ok(hits)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer(file, self)?;
        Ok(())
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
