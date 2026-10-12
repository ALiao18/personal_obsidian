_Team Members_
Jakob - math, CS minor
1. alignment, interpretability, RL
2. latent spaces + reasoning
3. on policy distillation

Adi - statistics, CS minor
1. alignment, interpretability, evaluating benchmarks of safe AI
2. alignment faking 
3. recommend latent reasoning

Andrew
1. international business BS, grad at northeastern in quant finance
2. research experience, research assistant for AI lab at school, curr research assistant for defense industrial institute. 
3. interpretability, quantitative finance

Abdel
1. undergrad stevens institute, CS and cybersecurity
2. founded 2 startups, worked
3. curr technical 
4. how deep alignment reaches after pretraining

_TODOs_
1. I will ask if any of the PIs have any topics for alignment, or try to propose a few myself. I also need to show some latent reasoning and mech interp papers
2. __Jakob and Adi__ will read up on alignment/internal representations
3. __Abdel and Andrew__ will focus on benchmarks and capabilities for LLMs

@channel Hi Team, here are some resources that could help your lit rev
_Jakob and Adi_: 
1. Important modern latent reasoning models. https://arxiv.org/pdf/2412.06769, https://arxiv.org/pdf/2502.21074. CODI was trained via self-distillation. Since these models reason in latent space, it could be interesting to look in their internals w/ mech interp techniques since we really have "no idea" how they think! One example of using mech interp on LRMs is https://openreview.net/pdf?id=EV30qkZXrR, from Algoverse. Personally, I think they overstate their scope but its a good example of a project probing internals of LRMs. I also have some ideas regarding latent reasoning that we can discuss and refine if needed.
_Abdel and Andrew_
2. "Beyond Chains of Thought: Benchmarking Latent-Space Reasoning Abilities in Large Language Models" (arxiv.org/abs/2504.10615)
3. PHYBench (arxiv.org/abs/2504.16074, NeurIPS 2025 Datasets & Benchmarks track). Physical reasoning

Some general good resources for mech interp
- https://arxiv.org/pdf/2501.16496 - survey paper on open problems for mech interp. Pretty long so I would recommend reading the introduction and finding 1-2 sub-chapters within the rest to dive deep once you find something interesting. Some keywords I think are good directions: _attribution graphs, cross-layer superposition, autointerp_, 
- https://medium.com/@joshuaplacidi/how-to-read-an-artificial-neural-brain-6e092f09b9f1 - intro to attribution graphs. Personally, I find this technique very interesting because it allows us to visually see how an LLM “thinks” and use the properties of graphs to test. 

I will update soon on PI projects. Please give me a short report on how the next meeting goes. Most importantly, communicate often on the slack channel and be active. Good work today!