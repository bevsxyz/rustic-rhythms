use rmp_serde::from_slice;
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
mod inference;
mod preprocess;
use crate::inference::get_top_k_similar;
use crate::preprocess::{Data, Embedding, reshape_flat_vec};
use web_sys::js_sys;


thread_local! {
    static EMBEDDING: RefCell<Option<Embedding>> = RefCell::new(None);
}

// Set normalized matrix from JS, store vector + dimensions
pub fn set_normalized_matrix(embedding: Embedding) {
    EMBEDDING.with(|cell| {
        *cell.borrow_mut() = Some(embedding);
    });
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

#[wasm_bindgen]
pub fn process_quantized_msgpack(msgpack_data: &[u8]) -> String {
    let data: Data = from_slice(msgpack_data).expect("Deserialization failed");
    let embedding = reshape_flat_vec(&data.weights);
    set_normalized_matrix(embedding);
    data.songs.dictionaries.title.get(1).expect("str").to_string()
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
    EMBEDDING.with(|cell| {
        let embedding_ref = cell.borrow();
        if let Some(embedding) = embedding_ref.as_ref() {
            let results = get_top_k_similar(embedding, query_idx, k);
            let array = js_sys::Array::new();

            for (idx, sim) in results {
                let res = SimilarityResult {
                    index: idx,
                    similarity: sim,
                };
                array.push(&JsValue::from(res));
            }
            Ok(array)
        } else {
            Err(JsValue::from_str("Normalized matrix not initialized"))
        }
    })
}
