This is a finetune of whisper-base.en created with to aim to improve the transcription performance on fantasy names and terms from the video game Skyrim.

# Performance
The finetuned model performs a lot better at transcribing Skyrim voicelines compared to the base model. 
Performance on general English drops slightly.

# Training Data

The model was finetuned on a dataset consisting of 85% Skyrim voicelines and 15% Common Voice English. 

From a list of target words & phrases I wanted the model to learn, voicelines from the base game and community mods which contain these words were choosen.

5344 Skyrim voicelines from various speakers were used for training. Each has an average length of ~7s

# Finetuning

My initial attempts always resulted in a decrease in performance on general English of from Whispers ~0.3 WER to ~0.37 on the Commonvoice Test set.

Freezing 3 of the decoder layers in addition to the encoder improved the performance on general English to 0.33 WER, while showing next to no impact on the Skyrim voicelines.