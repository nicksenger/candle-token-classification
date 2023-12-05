use candle_core::{Device, Error as E, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax_last_dim, Dropout, Linear, VarBuilder};
use tokenizers::Tokenizer;

use crate::{EntityGroup, BILOU};

use super::Config;

pub trait BertLikeModel: Sized {
    type Config<'a>: From<&'a Config>;

    fn load<'a>(vb: VarBuilder, config: &Self::Config<'a>) -> Result<Self>;
    fn device(&self) -> &Device;
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor>;
}

pub trait BertLikeTokenClassificationHead: Sized {
    type Model: BertLikeModel;

    fn new(device: Device, model: Self::Model, dropout: Dropout, classifier: Linear) -> Self;
    fn model(&self) -> &Self::Model;
    fn dropout(&self) -> &Dropout;
    fn classifier(&self) -> &Linear;

    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let vs = vb.pp("classifier");
        let model = Self::Model::load(vb.clone(), &config.into())?;
        let num_labels = config.id2label.len();
        let classifier_dropout = config
            .classifier_dropout
            .unwrap_or(config.hidden_dropout_prob);

        Ok(Self::new(
            model.device().clone(),
            model,
            Dropout::new(classifier_dropout),
            linear(config.hidden_size, num_labels, vs)?,
        ))
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let outputs = self.model().forward(input_ids, token_type_ids)?;
        let sequence_output = self.dropout().forward(&outputs, false)?;
        let logits = self.classifier().forward(&sequence_output)?;

        Ok(logits)
    }

    fn classify<'a>(
        &self,
        s: &'a str,
        labels: &[String],
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> Result<Vec<EntityGroup<'a>>> {
        let Ok(token_encoding) = tokenizer.encode(s, true) else {
            return Err(E::Msg("encoding".to_string()));
        };
        let offsets = token_encoding.get_offsets();
        let tokens = token_encoding.get_ids().to_vec();

        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let token_type_ids = input.zeros_like()?;
        let logits = self.forward(&input, &token_type_ids)?.squeeze(0)?;
        let scores = softmax_last_dim(&logits)?;
        let label_indices = scores.argmax(1)?;
        let v = label_indices.to_vec1::<u32>()?;

        let mut entity_groups = vec![];
        let mut entity_group_disagg: Vec<(Option<String>, usize, usize, BILOU)> = vec![];
        let mut last_bi: Option<BILOU> = None;

        // Aggregate neighboring tokens with the same label
        let mut bi = BILOU::I(Default::default());
        for i in 1..(v.len() - 1) {
            let token = tokenizer.id_to_token(tokens[i]);

            if token.as_ref().map(String::as_str) == Some("[SEP]") {
                break;
            }

            let (start, end) = offsets[i];

            bi = BILOU::from_entity_name(&labels[v[i] as usize]);
            if let Some(last_bi) = last_bi {
                if bi.tag() != last_bi.tag() || matches!(bi, BILOU::B(_)) {
                    let start = entity_group_disagg[0].1;
                    let end = entity_group_disagg[entity_group_disagg.len() - 1].2;
                    // entity_groups.push((&s[start..end], start, end, last_bi.tag().to_string()));
                    entity_groups.push(EntityGroup {
                        text: &s[start..end],
                        start,
                        end,
                        label: last_bi,
                    });
                    entity_group_disagg.clear();
                }
            }
            entity_group_disagg.push((token.clone(), start, end, bi.clone()));
            last_bi = Some(bi.clone());
        }

        if !entity_group_disagg.is_empty() {
            let start = entity_group_disagg[0].1;
            let end = entity_group_disagg[entity_group_disagg.len() - 1].2;
            entity_groups.push(EntityGroup {
                text: &s[start..end],
                start,
                end,
                label: bi,
            });
        }

        Ok(entity_groups)
    }
}
