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
