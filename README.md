# Candle Token Classification

This project provides token classification heads to use with pre-trained models of the BERT lineage (BERT, RoBERTa, DeBERTa, Electra, etc)

## Usage

```rust
use candle_token_classification::BertLikeTokenClassificationHead; // Import the token classifier trait from this library
use candle_token_classification::BertTokenClassificationHead; // Import the concrete classifier (BERT & ELECTRA are provided)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... retrieve Config, VarBuilder & Tokenizer from filesystem, HF hub, etc

    let classifier = BertTokenClassificationHead::load(vb, &config)?;

    use itertools::Itertools; // create an ordered list of labels for the chosen classifier
    let labels = config
        .id2label
        .iter()
        .sorted_by_key(|(i, _)| *i)
        .map(|(_, label)| label.to_string())
        .collect::<Vec<_>>();

    let output = classifier.classify( // classify some text (or use `classifier.forward` to get the output tensor)
        "This is the text we'd like to classify.",
        &labels,
        &tokenizer,
        &classifier.device
    );

    println!("{:?}", output) // view human-readable output or send for downstream processing, etc

    Ok(())
}
```

## Contributing

This repo was made specifically for my use-case of UPOS tagging with pre-trained models. As such it is likely missing things required to support work outside that scope. If you need other functionality, contributions are welcome :)
