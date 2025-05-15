use std::error::Error;

pub type Result<T> = core::result::Result<T, Box<dyn Error + Send + Sync>>;

pub trait TryDefault {
    fn default() -> Result<Self> where Self: Sized;
}