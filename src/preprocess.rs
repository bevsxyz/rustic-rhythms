use serde::Deserialize;

#[derive(Deserialize)]
pub struct QuantizedMatrix {
    pub quantized: Vec<u8>,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub zero_point: f32,
}

fn normalize_rows(row: &mut Vec<f32>) -> Vec<f32> {
    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        row.iter().map(|x| x / norm).collect()
    }
    else{
        row.to_vec()
    }
}

// Function to dequantize the u8 data to f32
fn dequantize(q: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    let row = q.iter()
         .map(|&v| (v as f32) * scale + zero_point)
         .collect();
    row
}

pub fn reshape_flat_vec(data: &QuantizedMatrix) -> Vec<f32> {
    let rows = data.shape[0];
    let cols = data.shape[1];
    let n_vectors = rows * cols;
    let mut matrix = Vec::with_capacity(n_vectors);
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let quantized = &data.quantized[start..end];
        let mut dequantized = dequantize(quantized, data.scale, data.zero_point);
        let normalized = normalize_rows(&mut dequantized);
        matrix.extend_from_slice(&normalized);
    }
    matrix
}

