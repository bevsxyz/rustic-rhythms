use wasm_bindgen::prelude::*;
use rmp_serde::from_slice;
use std::cell::RefCell;
mod preprocess;
mod inference;
use crate::inference::get_top_k_similar;
use crate::preprocess::{QuantizedMatrix, reshape_flat_vec};
use web_sys::js_sys;

thread_local! {
    static NORMALIZED_MATRIX: RefCell<Option<(Vec<f32>, usize, usize)>> = RefCell::new(None);
}

// Set normalized matrix from JS, store vector + dimensions
pub fn set_normalized_matrix(matrix: Vec<f32>, n_vectors: usize, embedding_dim: usize) {
    NORMALIZED_MATRIX.with(|cell| {
        *cell.borrow_mut() = Some((matrix, n_vectors, embedding_dim));
    });
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

#[wasm_bindgen]
pub fn process_quantized_msgpack(msgpack_data: &[u8]) -> Vec<i32> {
    let qm: QuantizedMatrix = from_slice(msgpack_data).expect("Deserialization failed");
    let weights = reshape_flat_vec(&qm);
    set_normalized_matrix(weights, qm.shape[0], qm.shape[1]);
    let v = vec![1, 2, 3];
    v
}

#[wasm_bindgen]
pub struct SimilarityResult {
    index: usize,
    similarity: f32,
}

#[wasm_bindgen]
impl SimilarityResult {
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> usize {
        self.index
    }

    #[wasm_bindgen(getter)]
    pub fn similarity(&self) -> f32 {
        self.similarity
    }
}

#[wasm_bindgen]
pub fn query_top_k_wasm(query_idx: usize, k: usize) -> Result<js_sys::Array, JsValue> {
    NORMALIZED_MATRIX.with(|cell| {
        if let Some((ref matrix, n_vectors, embedding_dim)) = *cell.borrow() {
            let results = get_top_k_similar(matrix, n_vectors, embedding_dim, query_idx, k);
            let array = js_sys::Array::new();

            for (idx, sim) in results {
                let res = SimilarityResult { index: idx, similarity: sim };
                array.push(&JsValue::from(res));
            }
            Ok(array)
        } else {
            Err(JsValue::from_str("Normalized matrix not initialized"))
        }
    })
}