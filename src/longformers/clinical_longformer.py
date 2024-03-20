from transformers import AutoTokenizer, AutoModelForMaskedLM, EvalPrediction, default_data_collator
from data.squad import Squad
from datasets import load_dataset, load_metric
from functools import partial
from .qa_model import QuestionAnsweringTrainer
from .qa_utils import postprocess_qa_predictions

def get_model():
    raw_datasets = load_dataset('data/squad.py')

    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")

    train_examples = raw_datasets["train"].select(range(1000))

    column_names = train_examples.column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    _prepare_validation_features = partial(prepare_validation_features, tokenizer=tokenizer, column_names=column_names)

    train_dataset = train_examples.map(_prepare_validation_features,num_proc=4,batched=True,remove_columns=column_names)

    _post_processing_function = partial(post_processing_function, answer_column_name=answer_column_name)

    trainer = QuestionAnsweringTrainer(model=model,tokenizer=tokenizer,post_process_function=_post_processing_function, compute_metrics=compute_metrics,data_collator=default_data_collator)

    results = trainer.predict(train_dataset, train_examples)
    metrics = results.metrics
    
    metrics["predict_samples"] = 1000

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)

def prepare_validation_features(examples, tokenizer, column_names, max_seq_length=384, doc_stride=128, pad_to_max_length=True):
    pad_on_right = tokenizer.padding_side == "right"
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length" if pad_to_max_length else False,
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def post_processing_function(examples, features, predictions, answer_column_name,stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            prefix=stage,
            n_best_size=20,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


if __name__=="__main__":
    get_model()