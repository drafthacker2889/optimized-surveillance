use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadwriteArray2}; // <--- IMPORT ReadWrite

/// FUSED KERNEL: Updates background AND calculates motion in a single CPU pass.
/// Complexity: O(N) | Memory Ops: 50% Reduction vs Python
#[pyfunction]
fn update_and_score(
    current_frame: PyReadonlyArray2<u8>,
    mut background_model: PyReadwriteArray2<f32>, // Mutable: We write to this
    learning_rate: f32,
    threshold: u8,
) -> PyResult<f32> {
    let current = current_frame.as_array();
    let mut bg = background_model.as_array_mut(); // Get mutable reference

    if current.shape() != bg.shape() {
        return Ok(0.0);
    }

    let mut changed_pixels = 0;
    let total_pixels = current.len();

    // The Magic: We iterate (Zip) over both images at the exact same time.
    // This keeps the CPU cache hot and prevents "cache misses".
    for (p_curr, p_bg) in current.iter().zip(bg.iter_mut()) {
        let pixel_val = *p_curr as f32;

        // 1. UPDATE BACKGROUND MODEL (The Math)
        // Formula: avg = (avg * (1 - alpha)) + (current * alpha)
        *p_bg = (*p_bg * (1.0 - learning_rate)) + (pixel_val * learning_rate);

        // 2. CALCULATE MOTION SCORE
        // We cast the updated float background back to u8 for comparison
        let bg_u8 = *p_bg as u8;
        
        // Calculate absolute difference manually
        let diff = if *p_curr > bg_u8 { *p_curr - bg_u8 } else { bg_u8 - *p_curr };

        if diff > threshold {
            changed_pixels += 1;
        }
    }

    Ok((changed_pixels as f32 / total_pixels as f32) * 100.0)
}

#[pymodule]
fn surveillance_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update_and_score, m)?)?;
    Ok(())
}