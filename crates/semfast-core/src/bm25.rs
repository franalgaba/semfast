use crate::chunking::ChunkId;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

const K1: f32 = 1.2;
const B: f32 = 0.75;
const MAX_POSTING_SCAN_RATIO: f32 = 0.05;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Bm25Index {
    document_count: usize,
    average_document_length: f32,
    document_lengths: BTreeMap<ChunkId, usize>,
    postings: BTreeMap<String, Vec<Posting>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Posting {
    chunk_id: ChunkId,
    term_frequency: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Bm25Hit {
    pub chunk_id: ChunkId,
    pub score: f32,
}

impl Bm25Index {
    pub fn build(chunks: impl IntoIterator<Item = (ChunkId, String)>) -> Self {
        let mut document_lengths = BTreeMap::new();
        let mut term_frequencies: HashMap<String, HashMap<ChunkId, usize>> = HashMap::new();

        for (chunk_id, text) in chunks {
            let tokens = tokenize(&text);
            document_lengths.insert(chunk_id, tokens.len());

            for token in tokens {
                let postings = term_frequencies.entry(token).or_default();
                *postings.entry(chunk_id).or_default() += 1;
            }
        }

        let document_count = document_lengths.len();
        let total_length = document_lengths.values().sum::<usize>();
        let average_document_length = if document_count == 0 {
            0.0
        } else {
            total_length as f32 / document_count as f32
        };

        let postings = term_frequencies
            .into_iter()
            .map(|(term, entries)| {
                let mut postings = entries
                    .into_iter()
                    .map(|(chunk_id, term_frequency)| Posting {
                        chunk_id,
                        term_frequency,
                    })
                    .collect::<Vec<_>>();
                postings.sort_by_key(|posting| posting.chunk_id);
                (term, postings)
            })
            .collect();

        Self {
            document_count,
            average_document_length,
            document_lengths,
            postings,
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<Bm25Hit> {
        if top_k == 0 || self.document_count == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<ChunkId, f32> = HashMap::new();

        for token in tokenize(query) {
            let Some(postings) = self.postings.get(&token) else {
                continue;
            };
            if self.is_high_frequency_term(postings.len()) {
                continue;
            }

            let inverse_document_frequency = self.inverse_document_frequency(postings.len());

            for posting in postings {
                let document_length = self
                    .document_lengths
                    .get(&posting.chunk_id)
                    .copied()
                    .unwrap_or_default() as f32;
                let term_frequency = posting.term_frequency as f32;
                let normalization = K1
                    * (1.0 - B
                        + B * document_length / self.average_document_length.max(f32::EPSILON));
                let score = inverse_document_frequency * (term_frequency * (K1 + 1.0))
                    / (term_frequency + normalization);
                *scores.entry(posting.chunk_id).or_default() += score;
            }
        }

        let mut hits = scores
            .into_iter()
            .map(|(chunk_id, score)| Bm25Hit { chunk_id, score })
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(top_k);
        hits
    }

    fn inverse_document_frequency(&self, document_frequency: usize) -> f32 {
        let numerator = self.document_count as f32 - document_frequency as f32 + 0.5;
        let denominator = document_frequency as f32 + 0.5;
        (1.0 + numerator / denominator).ln()
    }

    fn is_high_frequency_term(&self, document_frequency: usize) -> bool {
        if self.document_count < 1_000 {
            return false;
        }

        document_frequency as f32 / self.document_count as f32 > MAX_POSTING_SCAN_RATIO
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| {
            token
                .chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_matching_documents() {
        let index = Bm25Index::build([
            (ChunkId(1), "refund policy and return labels".to_string()),
            (ChunkId(2), "shipping estimates".to_string()),
        ]);

        let hits = index.search("refund", 1);

        assert_eq!(hits[0].chunk_id, ChunkId(1));
    }
}
