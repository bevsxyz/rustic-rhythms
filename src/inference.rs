use crate::preprocess::Embedding;
use std::cmp::Ordering;

// Compute dot product of two slices
fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

// Compute cosine similarity vector for a query vector against all rows of normalized matrix
fn compute_similarities(embedding: &Embedding, query_idx: usize) -> Vec<(usize, f32)> {
    let query_vec = &embedding.weights[query_idx * embedding.dim..(query_idx + 1) * embedding.dim];
    let mut similarities = Vec::with_capacity(embedding.n_vectors);
    for i in 0..embedding.n_vectors {
        let vec_i = &embedding.weights[i * embedding.dim..(i + 1) * embedding.dim];
        let similarity = dot_product(query_vec, vec_i);
        similarities.push((i, similarity));
    }
    similarities
}

// Exclude query index similarity by setting it to negative infinity
fn exclude_query_index(similarities: &mut [(usize, f32)], query_idx: usize) {
    if let Some(entry) = similarities.iter_mut().find(|(i, _)| *i == query_idx) {
        entry.1 = f32::NEG_INFINITY;
    }
}

// Get top k indices by similarity descending
fn get_top_k(similarities: &mut Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    similarities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarities.truncate(k);
    similarities.to_vec()
}

// Main function combining the above steps
pub fn get_top_k_similar(embedding: &Embedding, query_idx: usize, k: usize) -> Vec<(usize, f32)> {
    let mut similarities = compute_similarities(embedding, query_idx);
    exclude_query_index(&mut similarities, query_idx);
    get_top_k(&mut similarities, k)
}
