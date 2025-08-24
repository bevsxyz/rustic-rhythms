use wasm_bindgen::prelude::*;
use rmp_serde::from_slice;
use std::cell::RefCell;
use std::rc::Rc;
mod preprocess;
use crate::preprocess::{QuantizedMatrix, reshape_flat_vec};

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
pub fn process_quantized_msgpack(msgpack_data: &[u8]) -> Vec<f32> {
    let qm: QuantizedMatrix = from_slice(msgpack_data).expect("Deserialization failed");
    let weights = reshape_flat_vec(&qm);
    let n_vectors = qm.shape[0] * qm.shape[1];
    //set_normalized_matrix(weights, n_vectors, qm.shape[1])
    weights[..32].to_vec()
}
