use std::error;
use std::fmt::Display;


#[derive(Debug, Clone)]
/// Defines an error caused by the use of an incorrect index in one of the 
/// structures exchanged during a pipeline. This is an internal error that 
/// only occurs if there is a bug in the algorithms ensuring the consistency 
/// of this data, and it cannot be recovered from except by abandoning the 
/// ongoing process and report an error. However, it is preferred over panicking 
/// to ensure safe usage of the library (please document an issue in this case,
/// providing the message available in this struct). 
pub struct IndexError {
    message: String,
}

impl IndexError {
    pub fn new(array_desc: &str, index: usize) -> Self {
        Self {
            message: format!("error accessing index {index} in {array_desc}"),
        }
    }

    pub fn with(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl error::Error for IndexError { }

impl Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}