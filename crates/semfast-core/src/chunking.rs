use crate::metadata::Metadata;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ChunkId(pub u64);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub document_id: String,
    pub text: String,
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkingConfig {
    pub max_words: usize,
    pub overlap_words: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_words: 220,
            overlap_words: 32,
        }
    }
}

pub fn chunk_text(
    base_id: u64,
    document_id: &str,
    text: &str,
    metadata: &Metadata,
    config: ChunkingConfig,
) -> Vec<Chunk> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let max_words = config.max_words.max(1);
    let overlap_words = config.overlap_words.min(max_words.saturating_sub(1));
    let step = max_words - overlap_words;

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut ordinal = 0;

    while start < words.len() {
        let end = (start + max_words).min(words.len());
        chunks.push(Chunk {
            id: ChunkId(base_id + ordinal),
            document_id: document_id.to_string(),
            text: words[start..end].join(" "),
            metadata: metadata.clone(),
        });

        if end == words.len() {
            break;
        }

        start += step;
        ordinal += 1;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_ids_are_stable_and_incremental() {
        let chunks = chunk_text(
            42,
            "doc",
            "one two three four five",
            &Metadata::empty(),
            ChunkingConfig {
                max_words: 2,
                overlap_words: 0,
            },
        );

        let ids = chunks.iter().map(|chunk| chunk.id).collect::<Vec<_>>();

        assert_eq!(ids, vec![ChunkId(42), ChunkId(43), ChunkId(44)]);
    }
}
