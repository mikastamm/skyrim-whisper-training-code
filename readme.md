
# Whisper-base.en Finetune for Skyrim Terminology

This is a CTranslate2-format finetune of whisper-base.en that improves transcription of fantasy names and terms from Skyrim while maintaining general English accuracy.

Use as a drop-in replacement for [this base model](https://huggingface.co/guillaumekln/faster-whisper-base.en) in faster-whisper.

## Performance

- **Common Voice 11 test set WER**: 27.17 (Base model: 26.99)
- **Skyrim-specific terms error rate**: 20.49% (Base model: 50.82%)

## Training Data

- 85% Skyrim voicelines, 15% Common Voice English
- 5,344 Skyrim voicelines (average ~7s each)
- Voicelines selected to contain target fantasy words/phrases

## Finetuning Approach

Initial attempts decreased general English performance. Solution: freezing 3 decoder layers plus the encoder maintained Common Voice accuracy while significantly improving Skyrim term transcription.


# Cherry-Picked Results from Eval

| Source | Line |
| --- | --- |
| **ground truth** | **`Falkreath`'s banner bears the stag. They say it's because they'll never let a woman rule the hold.** |
| _this model_ | `Falkreath`'s banner bears the stag. They say it's because they'll never let a woman rule the hold. |
| whisper-base.en | `Fulkreet`'s banner bears the stag. They say it's because they'll never let a woman rule the hold. |
| **ground truth** | **Her daughter was sent to `Whiterun`, the skirmishes there have been violent. The `whiterun` `legate`, he needed to know the `Stormcloak` positions.** |
| _this model_ | `Hadata` was sent to `Whiterun`. The skirmishes there have been violent. The `Whiterun` `Legate`, he needed to know the `Stormcloak` positions. |
| whisper-base.en | Her daughter was sent to `White Run`. The skirmishes there have been violent. The `White Run` `laggot`, he needed to know the `storm clock` positions. |
| **ground truth** | **I never heard of `Tiber Septim` killing any dragons.** |
| _this model_ | I never heard of `Tiber Septim` killing any dragons. |
| whisper-base.en | I never heard of `Tiberceptom` killing any dragons. |
| **ground truth** | **Drink `Honningbrew Mead` with a `wench` on each arm? Ha ha!** |
| _this model_ | Dring `Honningbrew mead` with a `wench` on each arm? |
| whisper-base.en | During `Honning Broomade` with a `winch` on each arm? |
| **ground truth** | **`Riften, eh`? Probably down in the `Ratway`, then. It's where I'd go.** |
| _this model_ | `Riften-Ei`? Probably down in the `Ratway` then. It's where I'd go. |
| whisper-base.en | `Rift in A`. Probably down in the `rat way` then. It's where I'd go. |
| **ground truth** | **`Volunruud`? Well, that is interesting. I know this place...** |
| _this model_ | `Volenrud`? Well, that is interesting. I know this place. |
| whisper-base.en | `Volon Rood`? Well, that is interesting. I know this place. |
| **ground truth** | **He proposed that the real reason `Saarthal` was fought over by the Elves and the Nords was because something was unearthed there.** |
| _this model_ | He proposed that the real reason `Sauthal` was fought over by the elves and the Nords was because something was unearthed there. |
| whisper-base.en | He proposed that the real reason `Sothaw` was fought over by the elves and the Nords was because something was unearthed there. |
| **ground truth** | **Yes, but a `skeleton` has many bones, and they all look alike. A bone was stolen and switched with a `nameless one`, by those loyal to `Ysgramor`'s kin.** |
| _this model_ | Yes, but `Eskeletan` has many bones, and they all look alike. A bone was stolen, switched with a `Nameless One` by those loyal to `Ysgramor`'s kin. |
| whisper-base.en | Yes, but a `skeleton` has many bones, and they all look alike. A bone was stolen, switched with a nameless one by those loyal to `Isgrimor`'s kin. |
| **ground truth** | **Then you'll need to find `Elenwen`'s office and search her files. `Malborn` should be able to point you in the right direction.** |
| _this model_ | Then you'll need to find `Elenwen`'s office and search her files. `Malborn` should be able to point you in the right direction. |
| whisper-base.en | Then you'll need to find `Ellenwyn`'s office and search her files. `Malborn` should be able to point you in the right direction. |
| **ground truth** | **`Pa` was a sailor with a kid in every port from `Seyda Neen to Stros M'Kai`. One of them ports is `Windhelm`, and one of them kids is me.** |
| _this model_ | `Paar`'s a sailor with a kid in every putt from `Say the Name, Dostros Macai`. One of them ports is `Windhelm`, and one of them kids is me. |
| whisper-base.en | But I was a sailor with a kid in every port from `Saturday in Dosto, Macai`. One of them ports is `wind-hulme`. And one of them kids is me. |
| **ground truth** | **At least I have an excuse. I'm obligated to love my sister. `Leigelf` chose to.** |
| _this model_ | At least I have an excuse. I'm obligated to love my sister. `Legelf` chose too. |
| whisper-base.en | At least I have an excuse. I'm obligated to love my sister. The `gels` chose too. |
| **ground truth** | **`Izarn` says this place is no different than `Cidhna Mine`. The fool can say that because he's never been there.** |
| _this model_ | `Isan` says this place is no different than `Cidna Mine`. The fool can say that because he's never been there. |
| whisper-base.en | `Eason` says this place is no different than `Sid and mine`. The fool can say that because he's never been there. |
| **ground truth** | **`Boethiah`, `Mephala`, and `Azura`. When I sought to steal their relic, they chose my own `pupil` as their retribution.** |
| _this model_ | `Boethiah`, `Mephal` and the `Zul`. When I sought to steal their relic, they chose my own `pukule` as their retribution. |
| whisper-base.en | `Boethi`, `Mephal` and the `Zula`. When I sought to steal their relic, they chose my own `pupil` as their retribution. |
| **ground truth** | **`Vilkas`, take him out to the yard and see what he can do.** |
| _this model_ | `Vilkas`, take him out to the yard and see what he can do. |
| whisper-base.en | `Vilcus`, take him out to the yard and see what he can do. |