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

// Function to dequantize the u8 data to f32
fn dequantize(q: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    q.iter()
     .map(|&v| (v as f32) * scale + zero_point)
     .collect()
}

#[wasm_bindgen]
pub fn process_quantized_msgpack(msgpack_data: &[u8]) -> Vec<f32> {
    let qm: QuantizedMatrix = from_slice(msgpack_data).expect("Deserialization failed");
    dequantize(&qm.quantized, qm.scale, qm.zero_point)
}
