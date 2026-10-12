Soham Mehra

Current prosthetic limbs read EMG data, which comes from residual muscles to drive movement. For those with already atrophied little muscle, it doesn't make much sense. Morph tries to read motor intent from the brain via non-invasive, target EEG. 

_Tech:_ 512 electrode cap focused over motor cortex, to enable a 19 DoF target. Cap has 2-probe approach, where one electrode array sits on the scalp (signal and noise), a second one hovers above (noise), uses ML to handle the residual.
_Philosophy:_ Train on the noise, as preprocessing introduces human induced bias. 
_Long Term Goal:_ Brain foundation model (multi-modal neural dynamics across EEG, fNIRs, MEG, fMRI), as EEG has blind spots and different modalities should complement one another. 
# stances
1. Against the traditional preprocessing pipelines (bandpass filtering, ICA, CSP, etc.)
	1. This is reasonable, but data is the fundamental blocker. The only reason LLMs worked initially was because pre-training had such an immense amount of text data. EEG, not so much. 
2. a "brain foundation model" requires stitching together multiple modalities. Current approaches are by definition only single modality and not a true foundation model
	1. moreover, a model trained on all these modalities cannot expect the same during inference time, as nobody will be willing to put on all that when they want to use the BCI. It will have to run on just a few modalities and infer. 
3. scale won't solve subject variability as it did in language. 
	1. It makes sense, but we at least know some locality in the brain (motor cortex), true that signal-generation mapping is different, so supervised learning may not be the best approach
4. anti-invasive BCI
	1. partially disagree
5. start with EEG/fNIRS and train model that stitches them together
# Reading
1. [LARGE BRAIN MODEL FOR LEARNING GENERIC REPRESENTATIONS WITH TREMENDOUS EEG DATA IN BCI](https://arxiv.org/pdf/2405.18765)
	1. 
# Talk Points

# Question
1. How does multi-modal training even work? Stitching EEG/fNIRS together?
	1. paired multimodal datasets at scale
	2. architectures that can align modalities with incompatible timescales and measurement physics
	3. benchmarks that measure the quality of a learned representation
	4. transfer and domain adaptation across subjects
2. how can we solve the invariance issue?
3. Where can we read the work on Graph Fusion for Multimodal BCIs