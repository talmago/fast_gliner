use crate::model::input::tensors::gliclass::GLiClassExample;
use crate::util::result::Result;

pub(crate) const GLICLASS_LABEL_TOKEN: &str = "<<LABEL>>";
pub(crate) const GLICLASS_SEP_TOKEN: &str = "<<SEP>>";
pub(crate) const GLICLASS_EXAMPLE_TOKEN: &str = "<<EXAMPLE>>";

fn format_examples(examples: &[GLiClassExample]) -> Result<String> {
    if examples.is_empty() {
        return Ok(String::new());
    }

    let mut formatted = String::new();
    for example in examples {
        if example.text.is_empty() {
            return Err("invalid input: examples cannot contain an empty text".into());
        }
        if example.labels.is_empty() {
            return Err("invalid input: examples cannot contain an empty label list".into());
        }
        if example.labels.iter().any(|label| label.is_empty()) {
            return Err("invalid input: examples cannot contain an empty label".into());
        }

        formatted.push_str(GLICLASS_EXAMPLE_TOKEN);
        formatted.push_str(&example.text);
        formatted.push_str(" \nLabels:\n ");
        formatted.push_str(&example.labels.join(", "));
    }
    formatted.push_str(GLICLASS_SEP_TOKEN);
    Ok(formatted)
}

pub(crate) fn build_uniencoder_prompt(
    text: &str,
    labels: &[String],
    prompt_first: bool,
    task_prompt: Option<&str>,
    examples: &[GLiClassExample],
    sep_per_label: bool,
) -> Result<String> {
    let mut labels_and_sep = String::new();
    for label in labels {
        labels_and_sep.push_str(GLICLASS_LABEL_TOKEN);
        labels_and_sep.push_str(label);
        if sep_per_label {
            labels_and_sep.push_str(GLICLASS_SEP_TOKEN);
        }
    }
    if !sep_per_label {
        labels_and_sep.push_str(GLICLASS_SEP_TOKEN);
    }

    if let Some(task_prompt) = task_prompt.filter(|prompt| !prompt.is_empty()) {
        labels_and_sep.push_str(task_prompt);
    }

    let examples = format_examples(examples)?;
    let mut prompt = String::with_capacity(text.len() + labels_and_sep.len() + examples.len());
    if prompt_first {
        prompt.push_str(&labels_and_sep);
        prompt.push_str(text);
    } else {
        prompt.push_str(text);
        prompt.push_str(&labels_and_sep);
    }
    prompt.push_str(&examples);
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::build_uniencoder_prompt;
    use crate::model::input::tensors::gliclass::GLiClassExample;

    fn prompt(
        text: &str,
        labels: &[&str],
        prompt_first: bool,
        task_prompt: Option<&str>,
        examples: &[GLiClassExample],
    ) -> String {
        prompt_with(text, labels, prompt_first, task_prompt, examples, false)
    }

    fn prompt_with(
        text: &str,
        labels: &[&str],
        prompt_first: bool,
        task_prompt: Option<&str>,
        examples: &[GLiClassExample],
        sep_per_label: bool,
    ) -> String {
        build_uniencoder_prompt(
            text,
            &labels
                .iter()
                .map(|label| (*label).to_string())
                .collect::<Vec<_>>(),
            prompt_first,
            task_prompt,
            examples,
            sep_per_label,
        )
        .unwrap()
    }

    #[test]
    fn prompt_first_puts_labels_before_the_text() {
        let built = prompt(
            "Buy milk and eggs after work",
            &["shopping", "work", "personal"],
            true,
            None,
            &[],
        );

        assert_eq!(
            built,
            "<<LABEL>>shopping<<LABEL>>work<<LABEL>>personal<<SEP>>Buy milk and eggs after work"
        );
    }

    #[test]
    fn text_first_puts_labels_after_the_text() {
        assert_eq!(
            prompt("hello", &["a"], false, None, &[]),
            "hello<<LABEL>>a<<SEP>>"
        );
    }

    #[test]
    fn empty_task_prompt_leaves_the_basic_string_unchanged() {
        assert_eq!(
            prompt("hello", &["a"], true, Some(""), &[]),
            "<<LABEL>>a<<SEP>>hello"
        );
    }

    #[test]
    fn task_prompt_sits_after_the_separator() {
        assert_eq!(
            prompt(
                "The battery life is incredible",
                &["positive", "negative"],
                true,
                Some("Classify the sentiment:"),
                &[],
            ),
            "<<LABEL>>positive<<LABEL>>negative<<SEP>>Classify the sentiment:The battery life is incredible"
        );
        assert_eq!(
            prompt("hello", &["a"], false, Some("Do this:"), &[]),
            "hello<<LABEL>>a<<SEP>>Do this:"
        );
    }

    #[test]
    fn examples_are_appended_once_after_the_text() {
        let examples = vec![
            GLiClassExample {
                text: "Love this item, great quality!".to_string(),
                labels: vec!["positive".to_string(), "product".to_string()],
            },
            GLiClassExample {
                text: "Customer support was unhelpful".to_string(),
                labels: vec!["negative".to_string(), "service".to_string()],
            },
        ];

        assert_eq!(
            prompt(
                "Fast delivery",
                &["positive", "negative"],
                true,
                Some("Classify customer feedback:"),
                &examples,
            ),
            "<<LABEL>>positive<<LABEL>>negative<<SEP>>Classify customer feedback:Fast delivery<<EXAMPLE>>Love this item, great quality! \nLabels:\n positive, product<<EXAMPLE>>Customer support was unhelpful \nLabels:\n negative, service<<SEP>>"
        );
    }

    #[test]
    fn v4_prompt_puts_a_separator_after_every_label() {
        assert_eq!(
            prompt_with(
                "Buy milk and eggs after work",
                &["shopping", "work", "personal"],
                true,
                None,
                &[],
                true,
            ),
            "<<LABEL>>shopping<<SEP>><<LABEL>>work<<SEP>><<LABEL>>personal<<SEP>>Buy milk and eggs after work"
        );
        assert_eq!(
            prompt_with(
                "The battery life is incredible",
                &["positive", "negative"],
                true,
                Some("Classify the sentiment:"),
                &[],
                true,
            ),
            "<<LABEL>>positive<<SEP>><<LABEL>>negative<<SEP>>Classify the sentiment:The battery life is incredible"
        );
        assert_eq!(
            prompt_with("", &["yes", "no"], true, Some("Is this urgent?"), &[], true),
            "<<LABEL>>yes<<SEP>><<LABEL>>no<<SEP>>Is this urgent?"
        );
    }
}
