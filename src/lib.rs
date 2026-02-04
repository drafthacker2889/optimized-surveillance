use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

#[pyfunction]
fn calculate_motion_score(
    current_frame: PyReadonlyArray2<u8>,
    background_frame: PyReadonlyArray2<u8>,
    threshold: u8,
) -> PyResult<f32> {
    let current = current_frame.as_array();
    let background = background_frame.as_array();

    if current.shape() != background.shape() {
        return Ok(0.0);
    }

    let mut changed_pixels = 0;
    let total_pixels = current.len();

    // zip allows us to iterate two arrays at the same time
    for (p1, p2) in current.iter().zip(background.iter()) {
        // Rust allows subtracting references (&u8 - &u8) automatically
        // so 'diff' becomes a standard u8 number here.
        let diff = if p1 > p2 { p1 - p2 } else { p2 - p1 };
        
        // FIX: Removed the '*' because diff is already a number
        if diff > threshold {
            changed_pixels += 1;
        }
    }

    Ok((changed_pixels as f32 / total_pixels as f32) * 100.0)
}

#[pymodule]
fn surveillance_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_motion_score, m)?)?;
    Ok(())
}