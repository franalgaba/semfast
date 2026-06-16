use crate::chunking::ChunkId;
use crate::vector::{VectorHit, VectorIndex};
use crate::{Result, SemfastError};
use std::path::Path;
use turbovec::IdMapIndex;

const TURBOVEC_BIT_WIDTH: usize = 4;

pub struct TurboVecIndex {
    inner: IdMapIndex,
}

impl TurboVecIndex {
    pub fn build(vectors: &[f32], chunk_ids: &[ChunkId], dimensions: usize) -> Result<Self> {
        if dimensions == 0 {
            return Err(SemfastError::InvalidInput(
                "vector dimensions must be greater than zero".to_string(),
            ));
        }

        if vectors.len() != chunk_ids.len() * dimensions {
            return Err(SemfastError::InvalidInput(format!(
                "expected {} vector values for {} ids and {dimensions} dimensions, got {}",
                chunk_ids.len() * dimensions,
                chunk_ids.len(),
                vectors.len()
            )));
        }

        let ids = chunk_ids.iter().map(|id| id.0).collect::<Vec<_>>();
        let mut inner = IdMapIndex::new(dimensions, TURBOVEC_BIT_WIDTH)
            .map_err(|error| SemfastError::TurboVec(error.to_string()))?;
        inner
            .add_with_ids(vectors, &ids)
            .map_err(|error| SemfastError::TurboVec(error.to_string()))?;
        inner.prepare();

        Ok(Self { inner })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let inner = IdMapIndex::load(path)?;
        inner.prepare();
        Ok(Self { inner })
    }
}

impl VectorIndex for TurboVecIndex {
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<VectorHit>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let (scores, ids) = self.inner.search(query, top_k);
        Ok(scores
            .into_iter()
            .zip(ids)
            .map(|(score, id)| VectorHit {
                chunk_id: ChunkId(id),
                score,
            })
            .collect())
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.inner.write(path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_loads_and_searches_with_stable_ids() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("vectors.turbovec");
        let dimensions = 8;
        let vectors = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ids = [ChunkId(10), ChunkId(20)];
        let index = TurboVecIndex::build(&vectors, &ids, dimensions).unwrap();
        index.save(&path).unwrap();

        let loaded = TurboVecIndex::load(&path).unwrap();
        let hits = loaded.search(&vectors[..dimensions], 1).unwrap();

        assert_eq!(hits[0].chunk_id, ChunkId(10));
    }
}
