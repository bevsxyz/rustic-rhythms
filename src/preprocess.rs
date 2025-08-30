use serde::{Deserialize};

pub struct Embedding {
    pub weights: Vec<f32>,
    pub n_vectors: usize,
    pub dim: usize,
}

#[derive(Deserialize)]
pub struct Songs {
    pub titles: Vec<String>,
    artists: Vec<String>,
}

#[derive(Deserialize)]
pub struct Data {
    pub weights: QuantizedWeights,
    pub songs: Songs,
}

#[derive(Deserialize)]
pub struct QuantizedWeights {
    quantized: Vec<u8>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: f32,
}

fn normalize_rows(row: &mut Vec<f32>) -> Vec<f32> {
    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        row.iter().map(|x| x / norm).collect()
    } else {
        row.to_vec()
    }
}

// Function to dequantize the u8 data to f32
fn dequantize(q: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    let row = q.iter().map(|&v| (v as f32) * scale + zero_point).collect();
    row
}

pub fn reshape_flat_vec(data: &QuantizedWeights) -> Embedding {
    let rows = data.shape[0];
    let cols = data.shape[1];
    let mut weights = Embedding {
        weights: Vec::with_capacity(rows * cols),
        n_vectors: rows,
        dim: cols,
    };
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let quantized = &data.quantized[start..end];
        let mut dequantized = dequantize(quantized, data.scale, data.zero_point);
        let normalized = normalize_rows(&mut dequantized);
        weights.weights.extend_from_slice(&normalized);
    }
    weights
}
