use std::collections::{HashMap, HashSet};

pub struct RelationSchema {
    relations: HashMap<String, RelationSpec>,
}


impl RelationSchema {
    pub fn new() -> Self {
        Self { relations: HashMap::new() }
    }

    pub fn from_str(relations: &[&str]) -> Self {
        Self {
            relations: relations.iter().map(|r| (r.to_string(), RelationSpec::default())).collect()
        }
    }

    pub fn push(&mut self, relation: &str) {
        self.relations.insert(relation.to_string(), RelationSpec::default());
    }

    pub fn push_with_allowed_labels(&mut self, relation: &str, allowed_subjects: &[&str], allowed_objects: &[&str]) {
        self.relations.insert(relation.to_string(), RelationSpec::new(allowed_subjects, allowed_objects));
    }

    pub fn push_with_spec(&mut self, relation: &str, spec: RelationSpec) {
        self.relations.insert(relation.to_string(), spec);
    }

    pub fn relations(&self) -> &HashMap<String, RelationSpec> {
        &self.relations
    }
    
}

impl Default for RelationSchema {
    fn default() -> Self { Self::new() }
}

pub struct RelationSpec {
    allowed_subjects: Option<HashSet<String>>,
    allowed_objects: Option<HashSet<String>>,
}

impl RelationSpec {
    pub fn new(allowed_subjects: &[&str], allowed_objects: &[&str]) -> Self {
        Self {
            allowed_subjects: Some(allowed_subjects.iter().map(|x| x.to_string()).collect()),
            allowed_objects: Some(allowed_objects.iter().map(|x| x.to_string()).collect()),
        }
    }

    pub fn allows_subject(&self, label: &str) -> bool {
        match &self.allowed_subjects { None => true, Some(hs) => hs.contains(label) }
    }

    pub fn allows_object(&self, label: &str) -> bool {
        match &self.allowed_objects { None => true, Some(hs) => hs.contains(label) }
    }

    pub fn allows_one_of_subjects(&self, labels: &HashSet<String>) -> bool {
        match &self.allowed_subjects { None => true, Some(hs) => !hs.is_disjoint(labels) }
    }

    pub fn allows_one_of_objects(&self, labels: &HashSet<String>) -> bool {
        match &self.allowed_objects { None => true, Some(hs) => !hs.is_disjoint(labels) }
    }
}

impl Default for RelationSpec {
    fn default() -> Self {
        Self {
            allowed_subjects: None,
            allowed_objects: None,
        }
    }
}