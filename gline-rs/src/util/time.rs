use std::time::Instant;

use crate::util::result::Result;

pub fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub fn timed<T>(run: impl FnOnce() -> Result<T>) -> Result<(f64, T)> {
    let started = Instant::now();
    let value = run()?;
    Ok((elapsed_ms(started), value))
}
