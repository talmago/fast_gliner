//! Recursive schema and record assembly for GLiFormer structured extraction.
//!
//! A schema is an object, an array, or a scalar. Flat field lists and Pydantic
//! models are compiled to this tree before they reach the runtime. Record
//! anchors come from the structuring head. Nested objects are attached with
//! anchor-relation scores.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::util::result::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum SchemaNode {
    Scalar { type_name: String },
    Object { fields: Vec<(String, SchemaNode)> },
    Array { item: Box<SchemaNode> },
}

impl SchemaNode {
    pub fn scalar(type_name: impl Into<String>) -> Self {
        Self::Scalar {
            type_name: type_name.into(),
        }
    }

    pub fn object<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = (S, SchemaNode)>,
        S: Into<String>,
    {
        Self::Object {
            fields: fields
                .into_iter()
                .map(|(name, node)| (name.into(), node))
                .collect(),
        }
    }

    pub fn array(item: SchemaNode) -> Self {
        Self::Array {
            item: Box::new(item),
        }
    }

    pub fn has_nested_objects(&self) -> bool {
        match self {
            Self::Scalar { .. } => false,
            Self::Object { fields } => fields.iter().any(|(_, node)| node.has_nested_objects()),
            Self::Array { item } => {
                matches!(item.as_ref(), Self::Object { .. }) || item.has_nested_objects()
            }
        }
    }
}

/// Named record schemas for one `structure` call.
///
/// Each entry is one forward pass. The result object uses the same names, and
/// each value is a list of records.
#[derive(Debug, Clone, PartialEq)]
pub struct StructureSchema {
    pub fields: Vec<(String, SchemaNode)>,
}

impl StructureSchema {
    pub fn new<I, S>(fields: I) -> Self
    where
        I: IntoIterator<Item = (S, SchemaNode)>,
        S: Into<String>,
    {
        Self {
            fields: fields
                .into_iter()
                .map(|(name, node)| (name.into(), node))
                .collect(),
        }
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json).map_err(|err| err.to_string())?;
        let fields = value
            .get("fields")
            .and_then(Value::as_array)
            .ok_or("structure schema is missing a fields array")?;
        if fields.is_empty() {
            return Err("structure schema must contain at least one record".into());
        }
        let mut parsed = Vec::with_capacity(fields.len());
        for field in fields {
            let name = field
                .get("name")
                .and_then(Value::as_str)
                .ok_or("structure schema field is missing a name")?;
            if name.trim().is_empty() {
                return Err("structure schema record name cannot be empty".into());
            }
            let node = field
                .get("schema")
                .ok_or_else(|| format!("structure schema record `{name}` is missing a schema"))?;
            parsed.push((name.to_string(), parse_node(node)?));
        }
        validate_unique_names(&parsed)?;
        Ok(Self { fields: parsed })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    pub label: String,
    pub owner: Vec<String>,
    pub name: String,
    pub many: bool,
}

impl SchemaNode {
    pub fn scalar_fields(&self) -> Result<Vec<ScalarField>> {
        let mut fields = Vec::new();
        collect_scalars(self, &[], &mut fields)?;
        Ok(fields)
    }
}

#[derive(Debug, Clone)]
pub struct StructureSpan {
    pub label_index: usize,
    pub text: String,
    pub score: f32,
    pub start: usize,
}

/// Depth-first hierarchy pieces placed after the record name.
///
/// Scalar fields use the field token. Each nested object is wrapped in a child
/// marker and an end marker, matching the GLiFormer structuring prompt.
pub fn hierarchy_prompt(
    schema: &SchemaNode,
    field_token: &str,
    child_token: &str,
    end_token: &str,
) -> Result<Vec<String>> {
    let mut pieces = Vec::new();
    hierarchy_pieces(
        schema,
        &[],
        field_token,
        child_token,
        end_token,
        &mut pieces,
    )?;
    Ok(pieces)
}

fn hierarchy_pieces(
    node: &SchemaNode,
    owner: &[String],
    field_token: &str,
    child_token: &str,
    end_token: &str,
    pieces: &mut Vec<String>,
) -> Result<()> {
    let SchemaNode::Object { fields } = node else {
        return Err("a structure record must be an object".into());
    };
    for (name, child) in fields {
        let scalar = match child {
            SchemaNode::Scalar { .. } => true,
            SchemaNode::Array { item } => matches!(item.as_ref(), SchemaNode::Scalar { .. }),
            SchemaNode::Object { .. } => false,
        };
        if scalar {
            pieces.push(format!("{field_token} {}", path_label(owner, name)));
            continue;
        }
        let nested = match child {
            SchemaNode::Array { item } => item.as_ref(),
            SchemaNode::Object { .. } => child,
            SchemaNode::Scalar { .. } => continue,
        };
        pieces.push(format!("{child_token} {name}"));
        let mut next = owner.to_vec();
        next.push(name.clone());
        hierarchy_pieces(nested, &next, field_token, child_token, end_token, pieces)?;
        pieces.push(end_token.to_string());
    }
    Ok(())
}

pub fn assemble_structure(
    schema: &SchemaNode,
    fields: &[ScalarField],
    spans: &[StructureSpan],
    anchor_count: usize,
    membership: &[f32],
    objectness: &[f32],
    anchor_mask: &[f32],
    anchor_relations: Option<&[f32]>,
    threshold: f32,
) -> Result<Vec<Value>> {
    if schema.has_nested_objects() && anchor_relations.is_none() {
        return Err(
            "nested structure requires anchor relation scores from structuring.onnx; re-export the checkpoint with a multi-level structuring head"
                .into(),
        );
    }
    if spans.is_empty() || anchor_count == 0 {
        return Ok(Vec::new());
    }
    if objectness.len() < anchor_count || anchor_mask.len() < anchor_count {
        return Err("structuring objectness and anchor mask must cover every anchor".into());
    }
    let span_count = spans.len();
    if membership.len() < anchor_count * span_count {
        return Err("structuring membership does not cover every anchor and span".into());
    }
    if let Some(relations) = anchor_relations {
        if relations.len() < anchor_count * anchor_count {
            return Err("structuring anchor relations do not cover every anchor pair".into());
        }
    }

    let active = (0..anchor_count)
        .filter(|anchor| anchor_mask[*anchor] > 0.5 && sigmoid(objectness[*anchor]) > threshold)
        .collect::<Vec<_>>();

    let mut grouped: Vec<Vec<Vec<StructureSpan>>> =
        vec![vec![Vec::new(); fields.len()]; anchor_count];
    for (span_index, span) in spans.iter().enumerate() {
        let Some(field_index) = (span.label_index < fields.len()).then_some(span.label_index)
        else {
            continue;
        };
        let mut best: Option<(usize, f32)> = None;
        for anchor in &active {
            let score = sigmoid(membership[*anchor * span_count + span_index]);
            if score <= threshold {
                continue;
            }
            if best.map(|(_, current)| score > current).unwrap_or(true) {
                best = Some((*anchor, score));
            }
        }
        if let Some((anchor, _)) = best {
            grouped[anchor][field_index].push(span.clone());
        }
    }

    let mut anchors = Vec::new();
    for source in active {
        let mut owners = Vec::new();
        for (field, values) in fields.iter().zip(grouped[source].iter()) {
            if values.is_empty() || owners.iter().any(|owner| owner == &field.owner) {
                continue;
            }
            owners.push(field.owner.clone());
        }
        owners.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
        for owner in owners {
            let start = grouped[source]
                .iter()
                .zip(fields)
                .filter(|(_, field)| field.owner == owner)
                .flat_map(|(values, _)| values)
                .map(|span| span.start)
                .min()
                .unwrap_or(usize::MAX);
            let index = anchors.len();
            anchors.push(Anchor {
                index,
                source,
                owner,
                start,
            });
        }
    }

    let mixed_slots = anchors.iter().any(|anchor| {
        anchors
            .iter()
            .any(|other| other.index != anchor.index && other.source == anchor.source)
    });
    let parents = select_parents(
        &anchors,
        anchor_relations,
        anchor_count,
        threshold,
        mixed_slots,
    );
    let mut children_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (child, parent) in &parents {
        children_of.entry(*parent).or_default().push(*child);
    }

    let mut roots = anchors
        .iter()
        .filter(|anchor| anchor.owner.is_empty() && !parents.contains_key(&anchor.index))
        .map(|anchor| anchor.index)
        .collect::<Vec<_>>();
    roots.sort_by_key(|index| {
        anchors
            .iter()
            .find(|anchor| anchor.index == *index)
            .map(|anchor| anchor.start)
            .unwrap_or(usize::MAX)
    });

    Ok(roots
        .into_iter()
        .map(|anchor| {
            materialize(
                schema,
                &[],
                anchor,
                fields,
                &grouped,
                &anchors,
                &children_of,
            )
        })
        .collect())
}

struct Anchor {
    index: usize,
    source: usize,
    owner: Vec<String>,
    start: usize,
}

fn materialize(
    node: &SchemaNode,
    owner: &[String],
    anchor: usize,
    fields: &[ScalarField],
    grouped: &[Vec<Vec<StructureSpan>>],
    anchors: &[Anchor],
    children_of: &HashMap<usize, Vec<usize>>,
) -> Value {
    let Some(record) = anchors.iter().find(|candidate| candidate.index == anchor) else {
        return Value::Null;
    };
    let SchemaNode::Object {
        fields: object_fields,
    } = node
    else {
        return Value::Null;
    };
    let spans = &grouped[record.source];
    let mut object = Map::new();
    for (name, child) in object_fields {
        let value = match child {
            SchemaNode::Scalar { .. } => scalar_value(fields, spans, owner, name, false),
            SchemaNode::Array { item } if matches!(item.as_ref(), SchemaNode::Scalar { .. }) => {
                scalar_value(fields, spans, owner, name, true)
            }
            SchemaNode::Array { item } if matches!(item.as_ref(), SchemaNode::Object { .. }) => {
                let mut child_owner = owner.to_vec();
                child_owner.push(name.clone());
                let mut kids = children_of.get(&anchor).cloned().unwrap_or_default();
                kids.retain(|child_index| {
                    anchors.iter().any(|candidate| {
                        candidate.index == *child_index && candidate.owner == child_owner
                    })
                });
                kids.sort_by_key(|child_index| {
                    anchors
                        .iter()
                        .find(|candidate| candidate.index == *child_index)
                        .map(|candidate| candidate.start)
                        .unwrap_or(usize::MAX)
                });
                Value::Array(
                    kids.into_iter()
                        .map(|child_index| {
                            materialize(
                                item,
                                &child_owner,
                                child_index,
                                fields,
                                grouped,
                                anchors,
                                children_of,
                            )
                        })
                        .collect(),
                )
            }
            SchemaNode::Object { .. } => {
                let mut child_owner = owner.to_vec();
                child_owner.push(name.clone());
                materialize(
                    child,
                    &child_owner,
                    anchor,
                    fields,
                    grouped,
                    anchors,
                    children_of,
                )
            }
            SchemaNode::Array { .. } => Value::Array(Vec::new()),
        };
        object.insert(name.clone(), value);
    }
    Value::Object(object)
}

fn scalar_value(
    fields: &[ScalarField],
    spans: &[Vec<StructureSpan>],
    owner: &[String],
    name: &str,
    many: bool,
) -> Value {
    let Some(index) = fields
        .iter()
        .position(|field| field.owner == owner && field.name == name)
    else {
        return if many {
            Value::Array(Vec::new())
        } else {
            Value::Null
        };
    };
    let mut values = spans.get(index).cloned().unwrap_or_default();
    values.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then(right.score.total_cmp(&left.score))
    });
    if many {
        Value::Array(
            values
                .into_iter()
                .map(|span| Value::String(span.text))
                .collect(),
        )
    } else {
        values
            .into_iter()
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then(right.start.cmp(&left.start))
            })
            .map(|span| Value::String(span.text))
            .unwrap_or(Value::Null)
    }
}

fn select_parents(
    anchors: &[Anchor],
    relations: Option<&[f32]>,
    anchor_count: usize,
    threshold: f32,
    allow_text_fallback: bool,
) -> HashMap<usize, usize> {
    let Some(relations) = relations else {
        return HashMap::new();
    };
    let mut parents = HashMap::new();
    for child in anchors {
        if child.owner.is_empty() {
            continue;
        }
        let parent_owner = &child.owner[..child.owner.len() - 1];
        let mut best: Option<(usize, f32)> = None;
        for parent in anchors {
            if parent.index == child.index || parent.owner != parent_owner {
                continue;
            }
            let score = anchor_relation_score(parent, child, relations, anchor_count);
            if score < threshold {
                continue;
            }
            if best.map(|(_, current)| score > current).unwrap_or(true) {
                best = Some((parent.index, score));
            }
        }
        if best.is_none() && allow_text_fallback {
            let mut preceding: Option<(usize, f32, usize)> = None;
            for parent in anchors {
                if parent.index == child.index
                    || parent.owner != parent_owner
                    || parent.start >= child.start
                    || parent.start == usize::MAX
                {
                    continue;
                }
                let score = anchor_relation_score(parent, child, relations, anchor_count);
                let candidate = (parent.start, score, parent.index);
                let replace = preceding
                    .map(|current| {
                        candidate.0 > current.0
                            || (candidate.0 == current.0 && candidate.1 > current.1)
                            || (candidate.0 == current.0
                                && candidate.1 == current.1
                                && candidate.2 < current.2)
                    })
                    .unwrap_or(true);
                if replace {
                    preceding = Some(candidate);
                }
            }
            if let Some((_, _, parent)) = preceding {
                parents.insert(child.index, parent);
            }
            continue;
        }
        if let Some((parent, _)) = best {
            parents.insert(child.index, parent);
        }
    }
    parents
}

/// Relation scores are already probabilities. A slot that contains several
/// hierarchy levels becomes one record per level, and adjacent levels from
/// that slot are linked directly.
fn anchor_relation_score(
    parent: &Anchor,
    child: &Anchor,
    relations: &[f32],
    anchor_count: usize,
) -> f32 {
    let parent_of_child = &child.owner[..child.owner.len() - 1];
    if parent.source == child.source
        && parent.owner.as_slice() == parent_of_child
        && parent.start <= child.start
    {
        return 1.0;
    }
    relations[parent.source * anchor_count + child.source]
}

fn collect_scalars(node: &SchemaNode, owner: &[String], out: &mut Vec<ScalarField>) -> Result<()> {
    match node {
        SchemaNode::Scalar { .. } | SchemaNode::Array { .. } => {
            Err("a structure record must be an object".into())
        }
        SchemaNode::Object { fields } => {
            validate_object_fields(fields)?;
            for (name, child) in fields {
                match child {
                    SchemaNode::Scalar { .. } => out.push(ScalarField {
                        label: path_label(owner, name),
                        owner: owner.to_vec(),
                        name: name.clone(),
                        many: false,
                    }),
                    SchemaNode::Array { item }
                        if matches!(item.as_ref(), SchemaNode::Scalar { .. }) =>
                    {
                        out.push(ScalarField {
                            label: path_label(owner, name),
                            owner: owner.to_vec(),
                            name: name.clone(),
                            many: true,
                        })
                    }
                    SchemaNode::Array { item } => {
                        let mut next = owner.to_vec();
                        next.push(name.clone());
                        collect_scalars(item, &next, out)?;
                    }
                    SchemaNode::Object { .. } => {
                        let mut next = owner.to_vec();
                        next.push(name.clone());
                        collect_scalars(child, &next, out)?;
                    }
                }
            }
            Ok(())
        }
    }
}

fn path_label(owner: &[String], name: &str) -> String {
    if owner.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", owner.join("."), name)
    }
}

fn validate_object_fields(fields: &[(String, SchemaNode)]) -> Result<()> {
    if fields.is_empty() {
        return Err("structure object must contain at least one field".into());
    }
    let mut seen = Vec::new();
    for (name, _) in fields {
        if name.trim().is_empty() {
            return Err("structure field name cannot be empty".into());
        }
        if seen.iter().any(|existing: &String| existing == name) {
            return Err(format!("structure field `{name}` is duplicated").into());
        }
        seen.push(name.clone());
    }
    Ok(())
}

fn validate_unique_names(fields: &[(String, SchemaNode)]) -> Result<()> {
    let mut seen = Vec::new();
    for (name, node) in fields {
        if !matches!(node, SchemaNode::Object { .. }) {
            return Err(format!("structure `{name}` must be an object schema").into());
        }
        if seen.iter().any(|existing: &String| existing == name) {
            return Err(format!("structure `{name}` is duplicated").into());
        }
        seen.push(name.clone());
        node.scalar_fields()?;
    }
    Ok(())
}

fn parse_node(value: &Value) -> Result<SchemaNode> {
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .ok_or("structure schema node is missing a kind")?;
    match kind {
        "scalar" => {
            let type_name = value.get("type").and_then(Value::as_str).unwrap_or("str");
            Ok(SchemaNode::scalar(type_name))
        }
        "object" => {
            let fields = value
                .get("fields")
                .and_then(Value::as_array)
                .ok_or("structure object is missing fields")?;
            let mut parsed = Vec::with_capacity(fields.len());
            for field in fields {
                let name = field
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("structure object field is missing a name")?;
                let child = field
                    .get("schema")
                    .ok_or_else(|| format!("structure field `{name}` is missing a schema"))?;
                parsed.push((name.to_string(), parse_node(child)?));
            }
            Ok(SchemaNode::Object { fields: parsed })
        }
        "array" => {
            let item = value
                .get("item")
                .ok_or("structure array is missing an item schema")?;
            Ok(SchemaNode::array(parse_node(item)?))
        }
        other => Err(format!("unknown structure schema kind `{other}`").into()),
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(label: usize, text: &str, score: f32, start: usize) -> StructureSpan {
        StructureSpan {
            label_index: label,
            text: text.to_string(),
            score,
            start,
        }
    }

    fn employee_schema() -> SchemaNode {
        SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            ("company", SchemaNode::scalar("str")),
        ])
    }

    #[test]
    fn flat_anchors_become_separate_records() {
        let schema = employee_schema();
        let fields = schema.scalar_fields().expect("fields");
        let spans = vec![
            span(0, "Alice", 0.9, 0),
            span(1, "Acme", 0.8, 11),
            span(0, "Bob", 0.7, 20),
        ];
        let membership = vec![
            5.0, 5.0, -5.0, // anchor 0
            -5.0, -5.0, 5.0, // anchor 1
        ];
        let records = assemble_structure(
            &schema,
            &fields,
            &spans,
            2,
            &membership,
            &[5.0, 5.0],
            &[1.0, 1.0],
            None,
            0.5,
        )
        .expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["name"], "Alice");
        assert_eq!(records[0]["company"], "Acme");
        assert_eq!(records[1]["name"], "Bob");
        assert!(records[1]["company"].is_null());
    }

    #[test]
    fn nested_relations_attach_children_to_parents() {
        let schema = SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            (
                "departments",
                SchemaNode::array(SchemaNode::object([
                    ("name", SchemaNode::scalar("str")),
                    (
                        "employees",
                        SchemaNode::array(SchemaNode::object([
                            ("name", SchemaNode::scalar("str")),
                            ("role", SchemaNode::scalar("str")),
                        ])),
                    ),
                ])),
            ),
        ]);
        let fields = schema.scalar_fields().expect("fields");
        assert_eq!(
            fields
                .iter()
                .map(|field| field.label.as_str())
                .collect::<Vec<_>>(),
            vec![
                "name",
                "departments.name",
                "departments.employees.name",
                "departments.employees.role"
            ]
        );
        let spans = vec![
            span(0, "Acme", 0.9, 3),
            span(1, "Engineering", 0.9, 12),
            span(2, "Alice", 0.9, 30),
            span(3, "software engineer", 0.9, 37),
        ];
        let mut membership = vec![-5.0; 3 * 4];
        membership[0] = 5.0;
        membership[1 * 4 + 1] = 5.0;
        membership[2 * 4 + 2] = 5.0;
        membership[2 * 4 + 3] = 5.0;
        let mut relations = vec![-8.0; 9];
        relations[0 * 3 + 1] = 5.0;
        relations[1 * 3 + 2] = 5.0;
        let records = assemble_structure(
            &schema,
            &fields,
            &spans,
            3,
            &membership,
            &[5.0, 5.0, 5.0],
            &[1.0, 1.0, 1.0],
            Some(&relations),
            0.5,
        )
        .expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["name"], "Acme");
        assert_eq!(records[0]["departments"][0]["name"], "Engineering");
        assert_eq!(
            records[0]["departments"][0]["employees"][0]["name"],
            "Alice"
        );
        assert_eq!(
            records[0]["departments"][0]["employees"][0]["role"],
            "software engineer"
        );
    }

    #[test]
    fn mixed_anchor_splits_department_and_employee() {
        let schema = SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            (
                "departments",
                SchemaNode::array(SchemaNode::object([
                    ("name", SchemaNode::scalar("str")),
                    (
                        "employees",
                        SchemaNode::array(SchemaNode::object([
                            ("name", SchemaNode::scalar("str")),
                            ("role", SchemaNode::scalar("str")),
                        ])),
                    ),
                ])),
            ),
        ]);
        let fields = schema.scalar_fields().expect("fields");
        let spans = vec![
            span(0, "Acme", 0.9, 1),
            span(1, "Engineering", 0.9, 3),
            span(2, "Alice", 0.9, 5),
            span(3, "software engineer", 0.9, 8),
            span(2, "Bob", 0.9, 12),
            span(3, "designer", 0.9, 15),
            span(1, "Sales", 0.9, 17),
            span(2, "Carol", 0.9, 19),
            span(3, "account manager", 0.9, 22),
        ];
        let anchors = 4;
        let mut membership = vec![-5.0; anchors * spans.len()];
        let assign = [
            (0, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 6),
            (3, 7),
            (3, 8),
        ];
        for (anchor, span_index) in assign {
            membership[anchor * spans.len() + span_index] = 5.0;
        }
        let mut relations = vec![0.05; anchors * anchors];
        relations[0 * anchors + 1] = 0.9;
        relations[0 * anchors + 3] = 0.9;
        relations[1 * anchors + 2] = 0.9;
        let records = assemble_structure(
            &schema,
            &fields,
            &spans,
            anchors,
            &membership,
            &[5.0; 4],
            &[1.0; 4],
            Some(&relations),
            0.5,
        )
        .expect("records");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["name"], "Acme");
        let departments = records[0]["departments"].as_array().expect("departments");
        assert_eq!(departments.len(), 2);
        assert_eq!(departments[0]["name"], "Engineering");
        assert_eq!(departments[0]["employees"][0]["name"], "Alice");
        assert_eq!(departments[0]["employees"][0]["role"], "software engineer");
        assert_eq!(departments[0]["employees"][1]["name"], "Bob");
        assert_eq!(departments[0]["employees"][1]["role"], "designer");
        assert_eq!(departments[1]["name"], "Sales");
        assert_eq!(departments[1]["employees"][0]["name"], "Carol");
        assert_eq!(departments[1]["employees"][0]["role"], "account manager");
    }

    fn company_schema() -> SchemaNode {
        SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            (
                "departments",
                SchemaNode::array(SchemaNode::object([
                    ("name", SchemaNode::scalar("str")),
                    (
                        "employees",
                        SchemaNode::array(SchemaNode::object([
                            ("name", SchemaNode::scalar("str")),
                            ("role", SchemaNode::scalar("str")),
                        ])),
                    ),
                ])),
            ),
        ])
    }

    #[test]
    fn weak_relation_uses_text_order_only_after_a_mixed_slot() {
        let schema = company_schema();
        let fields = schema.scalar_fields().expect("fields");
        let spans = vec![
            span(0, "Acme", 0.9, 1),
            span(1, "Engineering", 0.9, 3),
            span(1, "Sales", 0.9, 17),
        ];
        let mut membership = vec![-5.0; 9];
        membership[0] = 5.0;
        membership[3 + 1] = 5.0;
        membership[6 + 2] = 5.0;
        let mut relations = vec![0.05; 9];
        relations[1] = 0.9;
        relations[2] = 0.2;
        let separate = assemble_structure(
            &schema,
            &fields,
            &spans,
            3,
            &membership,
            &[5.0; 3],
            &[1.0; 3],
            Some(&relations),
            0.5,
        )
        .expect("records");
        assert_eq!(
            separate[0]["departments"]
                .as_array()
                .expect("departments")
                .len(),
            1
        );

        let spans = vec![
            span(0, "Acme", 0.9, 1),
            span(1, "Engineering", 0.9, 3),
            span(2, "Alice", 0.9, 5),
            span(1, "Sales", 0.9, 17),
        ];
        let mut membership = vec![-5.0; 12];
        membership[0] = 5.0;
        membership[4 + 1] = 5.0;
        membership[4 + 2] = 5.0;
        membership[8 + 3] = 5.0;
        let mut relations = vec![0.05; 9];
        relations[1] = 0.9;
        relations[2] = 0.2;
        let recovered = assemble_structure(
            &schema,
            &fields,
            &spans,
            3,
            &membership,
            &[5.0; 3],
            &[1.0; 3],
            Some(&relations),
            0.5,
        )
        .expect("records");
        let departments = recovered[0]["departments"].as_array().expect("departments");
        assert_eq!(departments.len(), 2);
        assert_eq!(departments[0]["name"], "Engineering");
        assert_eq!(departments[0]["employees"][0]["name"], "Alice");
        assert_eq!(departments[1]["name"], "Sales");
    }

    #[test]
    fn nested_schema_without_relations_is_rejected() {
        let schema = SchemaNode::object([(
            "departments",
            SchemaNode::array(SchemaNode::object([("name", SchemaNode::scalar("str"))])),
        )]);
        let error = assemble_structure(&schema, &[], &[], 0, &[], &[], &[], None, 0.5)
            .expect_err("missing relations");
        assert!(
            error.to_string().contains("anchor relation scores"),
            "{error}"
        );
    }

    #[test]
    fn json_schema_matches_the_company_tree() {
        let schema = StructureSchema::from_json(
            r#"{
                "fields": [{
                    "name": "company",
                    "schema": {
                        "kind": "object",
                        "fields": [
                            {"name": "name", "schema": {"kind": "scalar", "type": "str"}},
                            {"name": "departments", "schema": {
                                "kind": "array",
                                "item": {"kind": "object", "fields": [
                                    {"name": "name", "schema": {"kind": "scalar", "type": "str"}}
                                ]}
                            }}
                        ]
                    }
                }]
            }"#,
        )
        .expect("schema");
        assert!(schema.fields[0].1.has_nested_objects());
        let labels = schema.fields[0].1.scalar_fields().expect("labels");
        assert_eq!(labels[0].label, "name");
        assert_eq!(labels[1].label, "departments.name");
    }

    #[test]
    fn hierarchy_prompt_wraps_nested_objects() {
        let schema = SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            (
                "departments",
                SchemaNode::array(SchemaNode::object([
                    ("name", SchemaNode::scalar("str")),
                    (
                        "employees",
                        SchemaNode::array(SchemaNode::object([
                            ("name", SchemaNode::scalar("str")),
                            ("role", SchemaNode::scalar("str")),
                        ])),
                    ),
                ])),
            ),
        ]);
        let pieces = hierarchy_prompt(&schema, "[FIELD]", "<<CHILD>>", "<<END>>").expect("prompt");
        assert_eq!(
            pieces,
            vec![
                "[FIELD] name",
                "<<CHILD>> departments",
                "[FIELD] departments.name",
                "<<CHILD>> employees",
                "[FIELD] departments.employees.name",
                "[FIELD] departments.employees.role",
                "<<END>>",
                "<<END>>",
            ]
        );
    }
}
