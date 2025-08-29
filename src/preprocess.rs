use serde::{Deserialize, Deserializer};

// Function to parse concatenated fixed-length strings from raw bytes
fn parse_fixed_length_strings(data: &[u8], str_len: usize) -> Vec<String> {
    data.chunks(str_len)
        .map(|chunk| {
            // Convert bytes to UTF-8 string, trim trailing null bytes
            String::from_utf8_lossy(chunk)
                .trim_end_matches('\0')
                .to_string()
        })
        .collect()
}

#[derive(Deserialize, Debug)]
pub struct Embedding {
    pub weights: Vec<f32>,
    pub n_vectors: usize,
    pub dim: usize,
}

fn deserialize_fixed_len_strings<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw_bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
    Ok(parse_fixed_length_strings(&raw_bytes, 8))
}

#[derive(Deserialize, Debug)]
pub struct Dictionaries {
    //#[serde(deserialize_with = "deserialize_fixed_len_strings")]
    pub title: Vec<String>, // concatenated string bytes (8 bytes each)
    //#[serde(deserialize_with = "deserialize_fixed_len_strings")]
    artist: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct EncodedColumns {
    title: Vec<u8>, // codes as raw bytes
    artist: Vec<u8>,
}

#[derive(Deserialize, Debug)]
pub struct Songs {
    pub dictionaries: Dictionaries,
    encoded_columns: EncodedColumns,
}

#[derive(Deserialize, Debug)]
pub struct Data {
    pub weights: QuantizedWeights,
    pub songs: Songs,
}

#[derive(Deserialize, Debug)]
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
