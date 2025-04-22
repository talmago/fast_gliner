use crate::util::result::Result;
use super::token::Token;
use regex::Regex;


/// Word-level tokenization
pub trait Splitter {
    fn split(&self, input: &str, limit: Option<usize>) -> Result<Vec<Token>>;
}


/// Word-level tokenization implemented using regular expressions
pub struct RegexSplitter {
    regex: Regex,
}


impl RegexSplitter {

    pub fn new(regex: &str) -> Result<Self> {
        Ok(Self {
            regex: Regex::new(regex)?
        })
    }

}

impl Default for RegexSplitter {
    fn default() -> Self {
        const DEFAULT_REGEX: &str = "\\w+(?:[-_]\\w+)*|\\S";
        Self::new(DEFAULT_REGEX).unwrap() // safe unwrap (as regex is const and correct)
    }
}


impl Splitter for RegexSplitter {

    fn split(&self, input: &str, limit: Option<usize>) -> Result<Vec<Token>> {
        let mut result = Vec::new();
        for m in self.regex.find_iter(input) {
            result.push(Token::new(m.start(), m.end(), m.as_str()));
            if let Some(limit) = limit {
                if result.len() >= limit {
                    break
                }
            }
        }
        Ok(result)
    }

}



#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_default_regex_splitter() -> Result<()> {
        let splitter = RegexSplitter::default();
        let tokens = splitter.split("This is an oh-yeah test", None)?;
        assert_eq!(tokens.len(), 5);
        let token = tokens.get(3).unwrap();
        assert_eq!(token.start(), 11);
        assert_eq!(token.end(), 18);
        assert_eq!(token.text(), "oh-yeah");
        Ok(())
    }

    #[test]
    fn test_unicode() -> Result<()> {
        let splitter = RegexSplitter::default();
        let tokens = splitter.split("Word with accents: éàèèçîù foo bar", None)?;
        assert_eq!(tokens.len(), 7);        
        Ok(())
    }

    #[test]
    fn test_limit() -> Result<()> {
        let splitter = RegexSplitter::default();
        let tokens = splitter.split("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10", Some(5))?;
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens.get(4).unwrap().text(), "w5");
        Ok(())
    }
}