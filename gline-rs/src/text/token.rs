/// A token with text and start/end offsets
#[derive(Debug)]
pub struct Token {
    start: usize,
    end: usize,
    text: String,
}


impl Token {
    pub fn new(start: usize, end: usize, text: &str) -> Self {
        Self { 
            start, end, 
            text: text.to_string() 
        }
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}




