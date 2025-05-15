/// Representation of a prompt as expected by GLiNER processing
#[derive(Debug)]
pub struct Prompt {
    prompt: Vec<String>,
    text_length: usize,    
    entities_length: usize,
}

impl Prompt {
    pub fn new(prompt: Vec<String>, text_length: usize, entities_length: usize) -> Self {
        Self { prompt, text_length, entities_length }
    }

    /// The actual prompt tokens
    pub fn tokens(&self) -> &Vec<String> {
        &self.prompt
    }

    /// Number of tokens in the text part
    pub fn text_len(&self) -> usize {
        self.text_length
    }

    /// Number of tokens in the entities part
    pub fn entities_len(&self) -> usize {
        self.entities_length
    }
}