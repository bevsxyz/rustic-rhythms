use wasm_bindgen::prelude::*;
use serde::Deserialize;
use rmp_serde::from_slice;

#[derive(Deserialize)]
pub struct QuantizedMatrix {
    quantized: Vec<u8>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: f32,
}

fn normalize_rows(row: Vec<f32>) -> Vec<f32> {
    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        row.iter().map(|x| x / norm).collect()
    }
    else{
        row
    }
}


// Function to dequantize the u8 data to f32
fn dequantize(q: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    let row = q.iter()
         .map(|&v| (v as f32) * scale + zero_point)
         .collect();
    let row= normalize_rows(row);
    row
}

fn reshape_flat_vec(data: &QuantizedMatrix) -> Vec<Vec<f32>> {
    let rows = data.shape[0];
    let cols = data.shape[1];
    let mut matrix = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        matrix.push(dequantize(&data.quantized[start..end], data.scale, data.zero_point));
    }
    matrix
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

#[wasm_bindgen]
pub fn process_quantized_msgpack(msgpack_data: &[u8]) -> Vec<f32> {
    let qm: QuantizedMatrix = from_slice(msgpack_data).expect("Deserialization failed");
    let weights = reshape_flat_vec(&qm);
    weights.first().expect("Expecting int").to_vec()
}
