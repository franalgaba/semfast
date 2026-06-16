use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Metadata {
    values: BTreeMap<String, MetadataValue>,
}

impl Metadata {
    pub fn new(values: BTreeMap<String, MetadataValue>) -> Self {
        Self { values }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.values.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.values.get(key)
    }

    pub fn matches_filter(&self, filter: &Filter) -> bool {
        filter.matches(self)
    }

    pub fn values(&self) -> &BTreeMap<String, MetadataValue> {
        &self.values
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Integer(i64),
    Boolean(bool),
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Filter {
    equals: BTreeMap<String, MetadataValue>,
}

impl Filter {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn equals(key: impl Into<String>, value: MetadataValue) -> Self {
        let mut equals = BTreeMap::new();
        equals.insert(key.into(), value);
        Self { equals }
    }

    pub fn and_equals(mut self, key: impl Into<String>, value: MetadataValue) -> Self {
        self.equals.insert(key.into(), value);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.equals.is_empty()
    }

    pub fn matches(&self, metadata: &Metadata) -> bool {
        self.equals
            .iter()
            .all(|(key, expected)| metadata.get(key).is_some_and(|actual| actual == expected))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_filter_matches_all_expected_values() {
        let mut metadata = Metadata::empty();
        metadata.insert("tenant", MetadataValue::String("acme".to_string()));
        metadata.insert("public", MetadataValue::Boolean(true));

        let filter = Filter::equals("tenant", MetadataValue::String("acme".to_string()))
            .and_equals("public", MetadataValue::Boolean(true));

        assert!(metadata.matches_filter(&filter));
    }
}
