use candle_core::{Device, Result, Tensor};
use candle_nn::{Dropout, Linear};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

use crate::{BertLikeModel, BertLikeTokenClassificationHead, Config};

impl From<&Config> for BertConfig {
    fn from(x: &Config) -> Self {
        serde_json::from_str(&serde_json::to_string(&x).expect("to json")).expect("from json")
    }
}

impl BertLikeModel for BertModel {
    type Config<'a> = BertConfig;

    fn load<'a>(vb: candle_nn::VarBuilder, config: &Self::Config<'a>) -> candle_core::Result<Self> {
        Self::load(vb, config)
    }

    fn device(&self) -> &candle_core::Device {
        &self.device
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        BertModel::forward(&self, input_ids, token_type_ids)
    }
}

pub struct TokenClassificationHead {
    model: BertModel,
    dropout: Dropout,
    classifier: Linear,
    pub device: Device,
}

impl BertLikeTokenClassificationHead for TokenClassificationHead {
    type Model = BertModel;

    fn new(
        device: Device,
        model: Self::Model,
        dropout: Dropout,
        classifier: Linear,
    ) -> Self {
        Self {
            model,
            dropout,
            classifier,
            device,
        }
    }

    fn model(&self) -> &Self::Model {
        &self.model
    }

    fn dropout(&self) -> &Dropout {
        &self.dropout
    }

    fn classifier(&self) -> &Linear {
        &self.classifier
    }
}
