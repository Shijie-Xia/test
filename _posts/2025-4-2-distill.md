---
layout: distill
title: Scaling reinforcement learning
description: an example of a distill-style blog post and main elements
tags: distill formatting
giscus_comments: true
date: 2021-05-22
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Shijie Xia
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton


bibliography: related.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Interactive Plots
  - name: Mermaid
  - name: Diff2Html
  - name: Leaflet
  - name: Chartjs, Echarts and Vega-Lite
  - name: TikZ
  - name: Typograms
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---




Recent research demonstrates that training LLMs through online reinforcement learning with rule-based rewards in tasks like mathematics and code can significantly enhance their reasoning abilities <d-cite key="deepseekai2025deepseekr1incentivizingreasoningcapability,MoonshotAI"></d-cite>. During the training process, models autonomously learn to master long-CoT test time scaling methods to solve challenging problems and demonstrate cognitive behaviors including self-reflection and self-correction. This phenomenon has been described as the RL scaling phenomenon<d-footnote>In the paper, we use "RL scaling" to describe the line of work.</d-footnote> or the "Aha moment." We systematically summarize recent works in Table 1. In the following sections, we detail the design considerations for each component.

## Training Algorithm

### REINFORCE

The REINFORCE <d-cite key="sutton1999policy"></d-cite> algorithm is a foundational policy gradient method in reinforcement learning that directly optimizes the expected return of a policy through gradient ascent. The algorithm optimizes the policy model $\pi_\theta$ by minimizing the loss:

$$ \mathcal{L}_{\rm REINFORCE}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=1}^T G_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right] $$

where $G_t$ is the discounted cumulative reward from time step $t$. Despite its simplicity, REINFORCE suffers from high variance in gradient estimates.

### Proximal Policy Optimization (PPO)

For the PPO algorithm <d-cite key="schulman2017proximalpolicyoptimizationalgorithms"></d-cite>, it optimizes the policy model by minimizing the loss:

$$ \mathcal{L}_{\rm PPO}(\theta) = - \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\rm old}}(O|q)}\frac{1}{|O|}\sum_{t=1}^{|O|}\min\left(\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{\rm old}}(o_t|q,o_{<t})}A_t,{\rm \text{clip}(\theta)}A_t\right) $$

$$ \text{clip}(\theta) = \text{clip}\left(\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{\rm old}}(o_t|q,o_{<t})},1-\varepsilon,1+\varepsilon\right) $$

where $\pi_\theta$ and $\pi_{\theta_{\rm old}}$ are the current and old policy models, and $q,o$ are the sampled questions and outputs. The $\text{clip}(\theta)$ function constrains policy updates to ensure stable training. $A_t$ is the advantage computed by applying GAE <d-cite key="schulman2018highdimensionalcontinuouscontrolusing"></d-cite> based on the rewards ${r_{\geq t}}$ and a learned value function $V_\psi$. The KL penalty can be added to the reward function:

$$ r_t = r_\varphi(q, o_{\leq t}) - \beta \log \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\rm ref}(o_t|q, o_{<t})} $$

where $r_\varphi$ is the reward model, $\pi_{\rm ref}$ is the reference model (initial SFT model), and $\beta$ is the coefficient of the KL penalty.

### Group Relative Policy Optimization (GRPO)

The GRPO algorithm <d-cite key="shao2024deepseekmathpushinglimitsmathematical"></d-cite> directly uses the average reward of multiple parallel sampled responses as the baseline, eliminating the need for additional value function approximation as in PPO. Specifically, for each question $q$, GRPO samples a group of outputs $\{o_1,o_2,\cdots,o_G\}$ from the old policy $\pi_{\theta_{\rm old}}$ and then optimizes the policy model $\pi_\theta$ by minimizing the loss:

$$ \mathcal{L}_{\rm GRPO}(\theta) = - \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} $$
$$ \frac{1}{G}\sum_{i=1}^G\frac{1}{|O_i|}\sum_{t=1}^{|O_i|}\left[\min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\rm old}}(o_{i,t}|q,o_{i,<t})}\hat{A_{i,t}},{\rm \text{clip}(\theta)}\hat{A_{i,t}}\right)-\beta\mathbb{D}_{\rm KL}[\pi_\theta||\pi_{\rm ref}]\right] $$

$$ \text{clip}(\theta) = \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\rm old}}(o_{i,t}|q,o_{i,<t})},1-\varepsilon,1+\varepsilon\right) $$

$$ \mathbb{D}_{\rm KL}[\pi_\theta||\pi_{\rm ref}] = \frac{\pi_{\rm ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log\frac{\pi_{\rm ref}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1 $$

where $\varepsilon$ and $\beta$ are hyper-parameters, and $\hat{A_{i,t}}$ is the advantage computed using a group of rewards corresponding to the outputs within each group.

### REINFORCE++

REINFORCE++ <d-cite key="hu2025reinforcesimpleefficientapproach"></d-cite> is a variant of the classical REINFORCE algorithm that integrates key optimization techniques from PPO while eliminating the need for a critic network. The algorithm incorporates several enhancements to address the limitations of REINFORCE as follows:

- It implements a token-level KL divergence penalty to prevent the policy from deviating too far from the initial model.
- It adopts PPO's clipping mechanism to constrain policy updates and maintain stability during training.
- It introduces mini-batch updates for improved training efficiency and better convergence rates.
- It employs comprehensive reward normalization and clipping to stabilize training by mitigating outliers and constraining reward values within predefined bounds.
- It implements advantage normalization using z-score normalization to ensure stable gradients and prevent divergence during training.

### Comparisons with different algorithms

We summarize the characteristics of different training algorithms in Table 2. Regarding computational budget, PPO shows predominant computational cost with four models to be loaded, among which the policy model and the critic model need to perform both inference and training. GRPO and REINFORCE++ eliminate the need for a critic model and achieve higher training stability than REINFORCE <d-cite key="hu2025reinforcesimpleefficientapproach"></d-cite>. Regarding performance, all algorithms except REINFORCE exhibit the RL scaling phenomenon. For specific performance comparisons, <d-cite key="hou2024doesrlhfscaleexploring"></d-cite> find that the performance of PPO and GRPO is similar in RLHF settings, while <d-cite key="xie2025logicrlunleashingllmreasoning"></d-cite> observe that the performance of PPO and REINFORCE++ is superior to GRPO in rule-based reward settings for synthetic logic puzzles. More rigorous and large-scale studies should be conducted to comprehensively evaluate the performance of these algorithms.

## Reward Function

The reward types can be categorized according to their source and granularity as follows:

- **Model-based reward**: In traditional RLHF <d-cite key="ouyang2022traininglanguagemodelsfollow"></d-cite> settings, an explicit reward model is learned from human preference data and guides the optimization process in RL training. The explicit reward model can be omitted by directly training on human preference data, resulting in an implicit reward model <d-cite key="rafailov2024directpreferenceoptimizationlanguage"></d-cite>.
- **Rule-based reward**: The term "rule-based" represents rewards that are well-defined and can be determined by explicit rules, sometimes also termed verifiable rewards. For example, for math problems with ground truth answers or code tasks with unit tests, response correctness can be easily verified and thus used to construct the reward. This can be further extended to include response format or language consistency. Even when verification is automated using a specialized model <d-cite key="chen2024huatuogpto1medicalcomplexreasoning,MoonshotAI"></d-cite>, we still attribute it to rule-based reward as long as the model's performance closely matches ideal rule verification.
- **Outcome reward**: In general settings, the rule-based reward or model-based reward is only given to the last token of response $o_i$, termed "outcome reward".
- **Process reward**: In multi-step reasoning tasks, the outcome reward may not be sufficient to supervise the policy model and help avoid logic errors in the solutions <d-cite key="shao2024deepseekmathpushinglimitsmathematical,lightman2023letsverifystepstep"></d-cite>. This necessitates more fine-grained rewards for each step, termed "process reward", which are typically calculated in a model-based way. Besides constructing process reward models, recent work also explores other ways to help achieve more accurate credit assignment. For example, <d-cite key="kazemnejad2024vineppounlockingrlpotential"></d-cite> replace the value networks in the PPO algorithm with unbiased Monte Carlo-based estimates. <d-cite key="hwang2024selfexploreenhancingmathematicalreasoning"></d-cite> and <d-cite key="setlur2024rlincorrectsyntheticdata"></d-cite> introduce MC-based methods to detect key errors in reasoning chains for use as ad-hoc mechanisms in DPO.

Figure 1 presents a comparison of different reward types. We detail the discussion below.

### Rule-based reward vs. model-based reward
The model-based reward can be applied to general tasks but also easily leads to reward hacking problems. This pipeline of constructing preference data to learn a reward model to proxy human preference can be applied to general tasks, leading to its widespread adoption. However, it has been observed that the reward is an imperfect proxy in the training process. There are two prevailing explanations for this phenomenon <d-cite key="rafailov2024scalinglawsrewardmodel"></d-cite>: 
1) OOD Robustness: the reward function is continuously queried using unseen model samples which are potentially out-of-distribution
2) Reward Mis-specification: Learned reward functions may exhibit spurious correlations that cause them to prefer unintended behaviors. 

These issues lead to reward overoptimization problems where, during the training process, while the proxy reward score monotonically increases, the golden reward score will saturate and then decrease <d-cite key="gao2022scalinglawsrewardmodel"></d-cite>. Although this issue can be alleviated by improving the reward model's capability through increased scale or training data <d-cite key="ouyang2022traininglanguagemodelsfollow,hou2024doesrlhfscaleexploring"></d-cite> or iteratively retraining the reward model to improve its supervision of the policy model <d-cite key="shao2024deepseekmathpushinglimitsmathematical"></d-cite>, the phenomenon still exists and hinders the success of large-scale RL <d-cite key="deepseekai2025deepseekr1incentivizingreasoningcapability"></d-cite>.

### Outcome reward vs. process reward
The fine-grained process reward may help improve the RL performance, but also introduces reward hacking problems as a model-based reward. Empirical results show that process rewards can help improve RL performance compared to using only outcome rewards <d-cite key="cui2024process,shao2024deepseekmathpushinglimitsmathematical"></d-cite>. However, it still faces several challenges: 
1) The construction of high-quality process rewards requires significant labor
2) An imperfect process reward model can be easily hacked. For example, <d-cite key="gao2024designingeffectiverlreward"></d-cite> find that repeating correct but unnecessary reasoning steps can lead to high rewards from process reward model. Although these issues can be addressed through reward refinement, it complicates the RL pipeline
3) Process rewards show less significant improvements in RL training than in parallel sampling settings. In parallel sampling settings, empirical results show that process reward models significantly outperform outcome reward models <d-cite key="lightman2023letsverifystepstep,wangMathShepherdVerifyReinforce2024b"></d-cite> in response selection. However, the gain is not as pronounced in RL settings <d-cite key="gaoInterpretableContrastiveMonte2024,cui2024process,shao2024deepseekmathpushinglimitsmathematical"></d-cite>.

### Optimization for rule-based reward
Rule-based rewards for eliciting long CoT reasoning primarily consist of correctness rewards and format rewards for specific tags. While this approach has proven sufficient for RL scaling, it can lead to potential content misalignment problems due to its narrow focus on accuracy. Two main issues arise from this approach. 

First, it may result in poor readability and inconsistent language use. Deepseek-R1 <d-cite key="deepseekai2025deepseekr1incentivizingreasoningcapability"></d-cite> addresses these challenges by initially fine-tuning their model on thousands of carefully selected long CoT examples. Additionally, it introduces a language consistency reward during RL training to mitigate language misalignment issues. 

Second, this approach can lead to excessive token length, potentially causing overthinking problems. To address this, Kimi k1.5 <d-cite key="MoonshotAI"></d-cite> implements length penalties in the later training stages, while T1 <d-cite key="hou2025advancinglanguagemodelreasoning"></d-cite> penalizes responses that either exceed the context window size or contain repetitive n-grams.

The success of RL in verifiable tasks demonstrates the importance of robust reward signals. As more research into RL scaling strengthens its theoretical and empirical foundation to facilitate implementation, it decouples the RL training process into two distinct steps: first defining verifiable rewards and then conducting RL training, as partially implemented in OpenAI's Reinforcement Fine-Tuning Service<d-footnote>https://openai.com/form/rft-research-program/</d-footnote>. Search-R1 <d-cite key="jin2025searchr1trainingllmsreason"></d-cite> utilizes a simple outcome reward function that verifies the correctness of final answers to conduct RL training and successfully endows LLMs with the ability to autonomously generate search queries during step-by-step reasoning with real-time retrieval, showcasing the power of RL beyond math and code. For future work in fields like open scientific questions, constructing reliable reward signals remains an open challenge and offers significant potential for innovation.

## Policy Model Selection

The policy model is a prerequisite for successful RL training. The selection criteria can be based on the following aspects:

### Model Family
As shown in Table 1, most RL scaling work utilizes Qwen2.5 as the base model. Recent studies demonstrate that Qwen2.5 exhibits cognitive behaviors such as verification and correction in its problem-solving process before applying RL <d-cite key="gandhi2025cognitivebehaviorsenableselfimproving,liu2025understanding,liu2025oatzero"></d-cite>, although the model cannot effectively use them. This indicates that the model's pretrained knowledge already contains these thinking patterns. <d-cite key="gandhi2025cognitivebehaviorsenableselfimproving"></d-cite> thoroughly investigate this phenomenon based on the observation that Qwen-2.5-3B exhibits substantial gains while Llama-3.2-3B quickly plateaus under identical RL training conditions for the game of Countdown. When Llama is primed with synthetic reasoning traces containing these behaviors or pretrained on cognitive behavioral augmentation data, it shows substantial improvements during RL, matching Qwen's performance trajectory. This highlights the importance of pretraining on corpus containing the cognitive behaviors before conducting RL.

### Model Size
While traditional RLHF settings show that larger models gain fewer benefits from reinforcement learning optimization <d-cite key="gao2022scalinglawsrewardmodel,hou2024doesrlhfscaleexploring"></d-cite>, RL scaling settings demonstrate that larger models achieve higher token efficiency and thus better performance <d-cite key="MoonshotAI"></d-cite>. The limited success in reproducing DeepSeek-R1-Zero's (671B) scaling behavior in 7B or smaller models for challenging tasks without long CoT cold start further suggests that model size significantly impacts scaling behavior.

## Training Data Construction

The quality and quantity of training data significantly affect the efficiency and upper bound of RL.

### Data Quality
Eliminating easy queries that require no further training helps save the unnecessary computation cost of RL as a post-training technique, where query difficulty can be estimated by sampling multiple times from the policy model to calculate the success rate for correct answers <d-cite key="MoonshotAI,chen2025empiricalstudyelicitingimproving"></d-cite>. Similarly, it is also beneficial to remove problems for which the current model lacks the fundamental capability to solve <d-cite key="chen2025empiricalstudyelicitingimproving"></d-cite>. From the training perspective, queries that the model consistently answers correctly or incorrectly introduce the gradient-decreasing problem. DAPO <d-cite key="yu2025dapoopensourcellmreinforcement"></d-cite> proposes a dynamic sampling strategy that over-samples and filters out prompts with accuracies of 1 and 0, observing significant performance gains, which can be considered an online difficulty control method.

### Data Quantity
In traditional RLHF settings, scaling the prompt quantity does not lead to significant performance improvements <d-cite key="hou2024doesrlhfscaleexploring"></d-cite>. However, this conclusion does not hold for RL scaling scenarios. Open-Reasoner-Zero <d-cite key="OpenReasonerZero2025"></d-cite> investigates the performance discrepancy between a 7.5K MATH training set and their curated 57K prompt set, finding that the larger set leads to continuous scaling in both accuracy and response length, while the smaller set plateaus. Similarly, DeepSeek-R1-Zero observes continuous performance improvements using their large-scale curated dataset <d-cite key="deepseekai2025deepseekr1incentivizingreasoningcapability"></d-cite>.

## Multi-stage Training

Training efficiency can be enhanced by employing the following multi-stage training strategy:

### Long CoT Cold Start
Fine-tuning on long CoT data before RL training can facilitate subsequent RL improvements <d-cite key="yeo2025demystifyinglongchainofthoughtreasoning"></d-cite> and mitigate early instability issues during RL training <d-cite key="deepseekai2025deepseekr1incentivizingreasoningcapability"></d-cite>. Additionally, enhancing the quality of long CoT significantly amplifies RL gains <d-cite key="yeo2025demystifyinglongchainofthoughtreasoning"></d-cite>. Furthermore, <d-cite key="li2025cold"></d-cite> demonstrates improved performance by incorporating sparse updates and adaptive termination mechanisms into the SFT loss function, which helps preserve response diversity after training.

### Iterative Lengthening Strategy
DeepScaleR-1.5B-Preview <d-cite key="deepscaler2025"></d-cite> initially restricts the context window size to 8K, during which the model generates shorter responses while training rewards increase. Upon reaching a critical point where model responses begin to lengthen, the context window size is expanded to 16K and subsequently to 24K (See the 'DeepScaleR-1.5B-Preview' row in Table 1). This strategy guides controlled response length expansion while reducing computational costs.
