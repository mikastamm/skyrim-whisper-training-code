# Eval 1: 

Tested on the unseen test set of skyrim voicelines extracted from the game. All speakers were unseen during training.

Checkpoint 1200

=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 11.91%
  Average Phrase Error Rate: 12.83%

Baseline Model (whisper-base.en):
  Average Overall WER: 18.73%
  Average Phrase Error Rate: 52.44%

Checkpoint: 600
=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 11.46%
  Average Phrase WER: 15.19%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 18.73%
  Average Phrase WER: 52.44%
  Untestable Voice Lines: 0

# Train 2: 10% Common Voice + 90% Skyrim (only 600 steps)

Finetuned Model:
  Average Overall WER: 26.54%
  Average Phrase Error: 39.16%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.46%
  Average Phrase Error: 65.81%
  Untestable Voice Lines: 0

# Train 3: 10% Common Voice + 90% Skyrim (1200 steps) + frozen encoder

Finetuned Model:
  Average Overall WER: 12.98%
  Average Phrase Error: 26.40%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.46%
  Average Phrase Error: 65.81%
  Untestable Voice Lines: 0

=== Evaluation Results on Common Voice English ===
base: 30.73%
checkpoint-1200: 33.96%

# Train 4: 0% Common Voice + 100% Skyrim (1200 steps) + frozen encoder


=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 12.83%
  Average Phrase Error: 25.79%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.46%
  Average Phrase Error: 64.45%
  Untestable Voice Lines: 0

=== Evaluation Results on Common Voice English ===
base: 30.73%
checkpoint-1200: 33.52%

# Train 5: 0% Common Voice + 100% Skyrim [ 2x Data (9218 Sample)]
=== Evaluation Results on Common Voice English ===
base: 30.73%
checkpoint-2500: 37.87%
checkpoint-2000: 37.75%
checkpoint-1500: 42.49%
checkpoint-1000: 38.73%

=== Aggregated Evaluation Results (2000 Steps) ===
Finetuned Model:
  Average Overall WER: 12.74%
  Average Phrase Error: 18.86%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 18.27%
  Average Phrase Error: 60.47%
  Untestable Voice Lines: 0

# Train 6: Same as above, but at least 3 occurences of each phrase to include (5241 Training Samples)

=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 14.90%
  Average Phrase Error: 15.17%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 56.72%
  Untestable Voice Lines: 0

=== Evaluation Results on Common Voice English ===
base: 30.73%
checkpoint-2000: 38.27%

# 10 % Commonvoice
=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 14.25%
  Average Phrase Error: 16.18%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 56.66%
  Untestable Voice Lines: 0

=== Evaluation Results on Common Voice English ===
base: 30.73%
checkpoint-900: 35.72%

# 15% Commonvoice
checkpoint-1200: 31.31%
=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 14.60%
  Average Phrase Error: 15.66%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 56.66%
  Untestable Voice Lines: 0
  
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=8,
        max_steps=TRAINING_STEPS,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=REPO_NAME,
        dataloader_num_workers=4
    )

# Same as above, traine
=== Evaluation Results on Common Voice English ===
checkpoint-2400: 34.63%

Finetuned Model:
  Average Overall WER: 14.50%
  Average Phrase Error: 15.38%
  Untestable Voice Lines: 29

# 3 Frozen decoder layers 
Checkpoint checkpoint-2400 WER: 32.93%

Finetuned Model:
  Average Overall WER: 14.01%
  Average Phrase Error: 14.99%
  Untestable Voice Lines: 29

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 51.28%
  Untestable Voice Lines: 29


Checkpoint checkpoint-1800 WER: 30.94%
=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 13.77%
  Average Phrase Error: 15.76%
  Untestable Voice Lines: 29

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 51.28%
  Untestable Voice Lines: 29

Checkpoint checkpoint-600 WER: 29.65%

=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 16.52%
  Average Phrase Error: 40.20%
  Untestable Voice Lines: 29

Baseline Model (whisper-base.en):
  Average Overall WER: 19.20%
  Average Phrase Error: 51.28%
  Untestable Voice Lines: 29

# Same as above, reselected data
checkpoint-1800: 29.14%

=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 13.86%
  Average Phrase Error: 20.49%
  Untestable Voice Lines: 18

Baseline Model (whisper-base.en):
  Average Overall WER: 18.46%
  Average Phrase Error: 50.82%
  Untestable Voice Lines: 18

# On all data
Base model WER: 26.99%
8-3FreezeDec2: 27.17%

# MORE DATA!!!!
=== Evaluation Results on Common Voice English ===
checkpoint-1800: 33.16%
Finetuned Model:
  Average Overall WER: 13.72%
  Average Phrase Error: 16.82%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.11%
  Average Phrase Error: 49.95%
  Untestable Voice Lines: 0


checkpoint-1200: 31.33%
=== Aggregated Evaluation Results ===
Finetuned Model:
  Average Overall WER: 14.31%
  Average Phrase Error: 21.74%
  Untestable Voice Lines: 0

Baseline Model (whisper-base.en):
  Average Overall WER: 19.11%
  Average Phrase Error: 49.95%
  Untestable Voice Lines: 0

  # sorendal/skyrim-whisper-small
Finetuned Model:
  Average Overall WER: 9.80%
  Average Phrase Error: 11.23%
  Untestable Voice Lines: 0

Commonvoice 15k 
Checkpoint skyrim-whisper-small WER: 23.63%

Baseline Model (whisper-base.en):
  Average Overall WER: 19.11%
  Average Phrase Error: 49.95%
  Untestable Voice Lines: 0



  # sorendal/skyrim-whisper-small-int8

