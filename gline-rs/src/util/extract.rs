use crate::util::result::Result;

/// Parses `name`, `name::str`, or `name::list`. A bare name is a list.
///
/// The bool is `true` when the spec asks for a single value (`::str`).
pub fn parse_field_spec(spec: &str) -> Result<(String, bool)> {
    let (field_name, single) = if let Some((name, suffix)) = spec.split_once("::") {
        let single = match suffix.trim() {
            "str" => true,
            "list" => false,
            other => {
                return Err(format!(
                    "invalid JSON schema field spec `{spec}`: unsupported type suffix `{other}` (expected `str` or `list`)"
                )
                .into())
            }
        };
        (name.trim().to_string(), single)
    } else {
        (spec.trim().to_string(), false)
    };

    if field_name.is_empty() {
        return Err(format!("invalid JSON schema field spec `{spec}`: empty field name").into());
    }

    Ok((field_name, single))
}

#[cfg(test)]
mod tests {
    use super::parse_field_spec;

    #[test]
    fn bare_name_is_a_list() {
        assert_eq!(
            parse_field_spec(" email ").unwrap(),
            ("email".to_string(), false)
        );
    }

    #[test]
    fn str_suffix_is_single() {
        assert_eq!(
            parse_field_spec("name::str").unwrap(),
            ("name".to_string(), true)
        );
    }

    #[test]
    fn list_suffix_is_many() {
        assert_eq!(
            parse_field_spec("skills::list").unwrap(),
            ("skills".to_string(), false)
        );
    }

    #[test]
    fn rejects_empty_and_unknown_suffix() {
        assert!(parse_field_spec("").is_err());
        assert!(parse_field_spec("::str").is_err());
        assert!(parse_field_spec("name::int")
            .unwrap_err()
            .to_string()
            .contains("int"));
    }
}
