mod config;
mod models;
mod traits;

pub use config::Config;
pub use models::bert::TokenClassificationHead as BertTokenClassificationHead;
#[cfg(feature = "electra")]
pub use models::electra::TokenClassificationHead as ElectraTokenClassificationHead;
pub use traits::{BertLikeModel, BertLikeTokenClassificationHead};

#[derive(Clone, Debug)]
/// Human-readable token classification output
pub struct EntityGroup<'a> {
    pub text: &'a str,
    pub start: usize,
    pub end: usize,
    pub label: BILOU,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BILOU {
    /// Beginning: The token is the beginning of an entity.
    B(String),
    /// Inside: The token is inside an entity.
    I(String),
    /// Last: The token is the last token of an entity.
    L(String),
    /// Outside: The token is outside any entity.
    O(String),
    /// Unit: The token is a single-token entity.
    U(String),
}

impl BILOU {
    pub fn from_entity_name(entity_name: &str) -> Self {
        if entity_name.starts_with("B-") {
            Self::B(entity_name.chars().skip(2).collect())
        } else if entity_name.starts_with("I-") {
            Self::I(entity_name.chars().skip(2).collect())
        } else {
            Self::I(entity_name.to_string())
        }
    }

    pub fn tag(&self) -> &str {
        match self {
            Self::B(s) | Self::I(s) | Self::L(s) | Self::O(s) | Self::U(s) => s,
        }
    }
}
