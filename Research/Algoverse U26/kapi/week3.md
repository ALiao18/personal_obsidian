Hey Team @Keshav Balakrishna @Abdulsalam Alshahrani @Ishika Aahna @Prashan Adhikari

Quick update connecting both your tracks. I looked into whether we could go straight from mech interp techniques to JEPA specifically, turns out most of the standard toolkit (logit lens, SAEs, activation patching) was built around having a discrete vocabulary and next-token prediction, and JEPA's predictor just doesn't have either. So a direct port from the LLM mech interp papers isn't quite ready yet.

However, JEPA's encoder is architecturally a vision transformer, and ViT/vision mech interp is its own subfield if you guys are down. Some reading for everyone:

[Laying the Foundations for Vision and Multimodal Mechanistic Interpretability & Open Problems](https://www.alignmentforum.org/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) (Joseph & Nanda) what's actually different between vision and language mech interp, plus a long open-problems list at the end if you want a sense of where the field's headed.

[Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) / [paper](https://arxiv.org/abs/2504.19475) TransformerLens-style activation caching, attention visualization, activation patching, and pretrained SAEs for CLIP and DINO you can just download and poke at.

[Interpreting CLIP's Image Representation via Text-Based Decomposition](https://arxiv.org/abs/2310.05916) (Gandelsman et al., ICLR 2024 Oral) concrete result. Decomposes CLIP's representation into patch/layer/head contributions and uses CLIP's paired text encoder to label them.

https://arxiv.org/pdf/2501.16496 - survey paper on open problems for mech interp. Pretty long so I would recommend reading the introduction and finding 1-2 sub-chapters within the rest to dive deep once you find something interesting. Some keywords I think are good directions: attribution graphs, cross-layer superposition, autointerp,

[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) (Darcet et al., ICLR 2024) Some "attention" in ViTs turns out to be artifact tokens repurposed for internal computation, not meaningful signal.

Don't try to read everything end to end, skim for what's interesting and go deep on that one. Feel free to propose some directions in the chat and I can help iterate. Happy to chat in the meantime.