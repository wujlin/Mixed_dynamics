# Phase Transitions in Collective Emotion: A Theoretical Framework Motivated by Media Dynamics During the COVID-19 Pandemic

# Abstract

Major societal crises, such as the COVID-19 pandemic, collective emotion can undergo abrupt and dramatic shifts, yet the underlying mechanisms driving these transitions remain poorly understood. This study introduces a theoretical framework, inspired by statistical physics, to model these phenomena as phase transitions. Motivated by the distinct roles of media during the COVID-19 pandemic, we propose a mixed-feedback model where competing mechanisms govern emotional dynamics. In our model, mainstream media provides a stabilizing negative feedback, while social media ("We-media") creates an amplifying positive feedback. We demonstrate that the interplay between these two forces acts as a control parameter for the entire system. As the influence of positive feedback grows, the system reaches a critical threshold and undergoes a second-order phase transition, shifting from a state of consensus to one of extreme polarization, which we term "emotional bubbles." Our analysis reveals characteristic features of this transition, including divergence of correlation length with a critical exponent  $\nu \approx 0.94$ . While motivated by a specific, large-scale event, our framework offers a powerful and broadly applicable new lens for quantitatively analyzing nonlinear dynamics in collective sentiment. It demonstrates how concepts from statistical physics can illuminate the emergence of systemic fragility in complex information ecosystems.

Keywords: Phase transitions, Collective emotion, Emotional bubbles, Mixed-feedback mechanisms, COVID-19

# Introduction

Understanding the abrupt, nonlinear transitions in collective human behavior—from opinion polarization and financial market crashes to social unrest—presents a formidable scientific challenge. While many models capture gradual shifts, they often fail to explain the sudden, system-wide changes that can emerge from complex social interactions  $^{1-5}$ . Such dynamics are frequently observed in online social networks  $^{6}$ , where adaptive feedback mechanisms, such as

influencer effects and algorithmic recommendations, can drive opinion clustering and create fragmented landscapes of transient consensus  $^{7-9}$ . These processes suggest that the architecture of our information ecosystems may contain mechanisms that push collective sentiment toward critical tipping points.

The COVID-19 pandemic provided a powerful real-world setting to study these complex dynamics on a global scale  $^{10-12}$ . The crisis triggered not only widespread negative emotions but also an "infodemic" of conflicting risk information, creating an unprecedented challenge for public mental health  $^{13-15}$ . Within this volatile environment, public sentiment did not always evolve predictably. Instead, it often displayed periods of apparent stability punctuated by rapid, dramatic shifts toward extreme polarization, a phenomenon we conceptually term "emotional bubbles." This pattern mirrors financial bubbles: initial stability masks underlying fragility. When information environments become imbalanced and critical thresholds are exceeded, collective emotions can shift abruptly from calm to collapse.

This observation motivates a central puzzle. Existing frameworks, such as the Social Amplification of Risk Framework  $^{16}$ , and empirical studies have identified numerous factors in emotional escalation, including political discontent  $^{17}$ , moral outrage  $^{18,19}$ , and real-time social media interactions  $^{20-22}$ . However, these approaches predominantly focus on linear amplification processes driven by self-reinforcing positive feedback. Our analysis of social media data from the COVID-19 pandemic in China revealed a more complex pattern that challenges these linear models: mainstream media exposure was associated with an inverted U-shaped relationship with emotional polarization, suggesting a stabilizing or moderating effect, whereas "We-media"(i.e., independent social media accounts) exposure showed a direct linear correlation with its escalation. This observation of seemingly opposing dynamics from different information sources cannot be fully explained by existing single-factor amplification models  $^{23}$ .

To explore a potential mechanism underlying this puzzle, we develop and analyze a novel theoretical framework grounded in statistical physics (See Figure 1). We propose that the competition between stabilizing negative feedback (analogous to mainstream media) and amplifying positive feedback (analogous to We-media) is a sufficient condition to produce the kind of abrupt, nonlinear transitions observed in collective emotion. Our goal is not to present an empirical validation of a model for all crises, but rather to investigate the rich, emergent behaviors that arise from a minimal set of competing feedback rules inspired by real-world observations. By applying phase transition analysis—a well-established method for studying critical behavior in physical systems—we can identify the precise conditions under which an information ecosystem can be tipped from a resilient state into a fragile one.

Our theoretical approach employs mean-field equations with psychological thresholds (see Methods for  $\phi$  and  $\theta$  parameter definitions) that govern individual emotional responses to

risk information. As psychological sensitivity increases, individuals engage in selective exposure behaviors that systematically erode moderate emotional states, creating conditions for critical transitions. This framework establishes emotional bubbles as threshold-mediated phenomena where competing feedback mechanisms determine whether systems exhibit continuous or discontinuous responses.

This paper is structured as follows. We first briefly present the empirical findings that motivate our theoretical inquiry. Next, we formally introduce our mixed-feedback model. We then conduct a detailed theoretical analysis of the model, using both mean-field equations and agent-based network simulations, to map its phase diagram and characterize its critical phenomena. Finally, we discuss the broader implications of our theoretical framework for understanding systemic fragility in information ecosystems and other complex social systems.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/e3c2c1aca0600a1961ac3fe806a09066055121e5f354695669093f6278711e6e.jpg)  
Figure 1 | Research design and methodological framework. Systematic approach to investigating emotional bubble formation through phase transition analysis. Empirical data reveals differential feedback patterns that motivate theoretical framework development. Phase transition analysis of the competing feedback system identifies critical behaviors with second-order and first-order transitions. Agent-based network simulations provide independent validation across parameter space. This framework demonstrates how phase transition analysis complements existing approaches to understand critical behaviors in emotional systems.

# Results

# Empirical Motivation: Competing Media-Sentiment Dynamics on Weibo During the COVID-19 Pandemic

To establish an empirical grounding for our theoretical exploration, we analyzed sentiment and media dynamics using three datasets from Weibo, China's largest microblogging platform, spanning the COVID-19 pandemic (January 2020 - December 2023). These datasets include 49,288 posts from verified "We-media" accounts (opinion-driven content), 36,480 posts from verified mainstream media outlets (authoritative sources), and 208,666 posts from individual users, collectively capturing a broad spectrum of the information ecosystem.

To ensure reliable content analysis, we developed and validated a multi-task machine learning framework to classify emotional states and risk perceptions. Our DistilBERT-based classifier achieved high performance across 100 independent training experiments, with emotion classification reaching  $98.0\%$  test accuracy (CV:  $0.73\%$ ) and risk classification for mainstream and We-media posts attaining  $91.5\%$  and  $96.7\%$  accuracy, respectively. The robustness of these classifications provides confidence in the data underlying our analysis (see Supplementary Figure 1 and Figure 2 for full performance metrics).

For this analysis, we defined "risk information" based on media reporting tendencies regarding the potential health consequences of Long COVID. For instance, reports minimizing long-term effects were classified as no-risk, while those emphasizing severe persistent symptoms were classified as risk. We focused our temporal analysis on December 2022, a period corresponding to the peak of Long COVID discussions and significant policy shifts in China, providing a rich context for observing media-sentiment interactions  $^{24}$ . Data were aggregated daily to capture the interplay between information exposure and collective emotional responses at the population level.

Our analysis of these data revealed several distinct and intriguing patterns. First, the temporal evolution of risk information and public sentiment showed that while risk reporting from both media types fluctuated, We-media content was more volatile and showed a significant positive Pearson correlation with high-arousal public emotion (anger, sarcasm) ( $r = 0.387$ ,  $p = 0.042$ ). In contrast, mainstream media reporting had no significant linear correlation with emotional states ( $r = 0.036$ ,  $p = 0.854$ ), suggesting different underlying dynamics (Figure 2a).

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/528d76919b1e860ab4faa4647303abdfa89b721a06fa2535117b2227a769e999.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/78a4d666bb3acb6a7b607bc3665938edec533c88f3bf02322a1f8054ce35780a.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/6026a58b23d3822b7b48c9104abca83bd1c1aaee1c611f07ecb300f8bfd363af.jpg)  
Figure 2 | Competing Media-Sentiment Dynamics on Weibo During the COVID-19 Pandemic. a, Temporal evolution of risk information from mainstream media (purple dotted) and We-media (red dotted), alongside high-arousal public emotion (orange dashed). b, A U-shaped relationship is observed between total risk exposure and emotional concentration. Shaded areas represent  $95\%$  confidence intervals. c, Mainstream media exposure exhibits an inverted U-shaped relationship with emotional polarization. d, We-media exposure shows a positive linear relationship with emotional polarization.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/472241394c1df042549dfcef314886fcbdec70022c5eed4e90d52e8716855fad.jpg)

Second, we examined the relationship between the total volume of risk information and the degree of emotional clustering, a metric we term emotional concentration (see Eq. 10). We observed a pronounced U-shaped relationship (Figure 2b), indicating that emotional clustering was highest when risk information was either very scarce or overly abundant, and lowest at moderate levels of exposure. Consistent with previous research on digital echo chambers and emotional contagion dynamics  $^{25,26}$ , this empirical finding provides validation that balanced information maintain emotional diversity, while extreme information systematically facilitate clustering through threshold-mediated mechanisms.

Third, by disaggregating the media sources, we found starkly different patterns. Mainstream media exposure exhibited a significant inverted U-shaped relationship with emotional polarization (see Eq. 11), with emotional polarization peaked at moderate levels of mainstream media exposure (Figure 2c).

Finally, We-media exposure showed a contrasting and simpler dynamic: a robust positive linear relationship with emotional polarization (Figure 2d). This pattern indicates that as

exposure to We-media content increased, so did the prevalence of extreme emotional states in the public discourse.

Taken together, these empirical findings present a puzzle. The public emotional landscape responds to information in a highly nonlinear fashion, and different media sources appear to exert opposing influences—one stabilizing or moderating, the other consistently amplifying. These observations of competing dynamics cannot be readily explained by simple, linear amplification models and thus provide the primary motivation for developing the theoretical framework presented in the next section.

# A Theoretical Model of Competing Feedback in Information Ecosystems

Motivated by the empirical puzzle of competing media-sentiment dynamics, we developed a theoretical model, depicted schematically in Figure 3, to investigate whether the interplay between stabilizing and amplifying feedback is a sufficient mechanism to produce such nonlinear collective behavior.

# The Model's Core Components

The model describes a population of individuals who can exist in one of three emotional states: high-arousal  $(X_{H})$ , medium-arousal  $(X_{M})$ , or low-arousal  $(X_{L})$ , where  $X_{i}$  represents the fraction of the population in state  $i$ , such that the total population is conserved:

$$
X _ {H} + X _ {M} + X _ {L} = 1 \backslash (1 \backslash)
$$

An individual's transition between these states is governed by their psychological sensitivity to risk information, which we parameterize using two thresholds: a low-arousal threshold,  $\phi$ , and a high-arousal threshold,  $\theta$  (where  $0 \leq \phi < \theta \leq 1$ ). Here,  $\phi$  represents the minimum proportion of risk signals required to move an individual out of a low-arousal state, while  $\theta$  is the proportion needed to trigger a high-arousal state. The interval  $\theta - \phi$  thus represents the buffer for moderate emotional responses; a smaller interval indicates higher public sensitivity to risk information.

For an individual connected to  $k$  media sources, the probability of receiving  $s$  risk signals follows a binomial distribution, where  $p_{\text{risk}}$  is the overall probability of any given signal being 'risk'. An individual transitions to a high-arousal state if the fraction of risk signals they receive,  $s / k$ , exceeds their threshold  $\theta$ . They transition to a low-arousal state if  $s / k$  is below  $\phi$ . The population-level fractions are then calculated by averaging over the degree distribution  $P(k)$ :

$$
X _ {H} = \sum_ {k} \square P (k) \sum_ {s = \left\lceil \theta k \right\rceil} ^ {k} \binom {k} {s} p _ {\text {r i s k}} ^ {s} \left(1 - p _ {\text {r i s k}}\right) ^ {k - s} \backslash (2 \backslash)
$$

$$
X _ {L} = \sum_ {k} \square P (k) \sum_ {s = 0} ^ {| \phi k |} \binom {k} {s} p _ {r i s k} ^ {s} \left(1 - p _ {r i s k}\right) ^ {k - s} \backslash (3 \backslash)
$$

where we assume  $P(k)$  follows a Poisson distribution. The fraction in the medium-arousal state is determined by the conservation constraint,  $X_{M} = 1 - X_{H} - X_{L}$ .

# The Competing Feedback Mechanism

The core of our model is the mechanism by which the overall risk probability,  $p_{\text{risk}}$ , is determined by the collective emotional state of the population, creating feedback loops. Inspired by our empirical observations, we define two distinct feedback mechanisms from two types of media sources.

First, mainstream media provides stabilizing negative feedback. Its propensity to report risk is designed to counteract the dominant public emotion: it decreases as high-arousal states ( $X_H$ ) rise and increases as low-arousal states ( $X_L$ ) rise. This is captured by:

$$
p _ {r i s k} ^ {\text {m a i n s t r e a m}} = \frac {1 - X _ {H} + X _ {L}}{2} \backslash (4 \backslash)
$$

Second, We-media provides amplifying positive feedback. Its risk reporting is driven by and reinforces the high-arousal state, reflecting an attention-based dynamic where sensational content is amplified:

$$
p _ {r i s k} ^ {\text {w e m e d i a}} = X _ {H} \backslash (5 \backslash)
$$

# System Coupling and the Control Parameter

The total probability of risk information in the ecosystem,  $p_{\text{risk}}$ , is the weighted average of these two sources. This formulation allows us to introduce a key external control parameter,  $r$ , which represents the fraction of mainstream media sources removed from the system:

$$
p _ {r i s k} = \frac {p _ {r i s k} ^ {\text {m a i n s t r e a m}} \cdot n _ {m} \cdot (1 - r) + p _ {r i s k} ^ {\text {w e m e d i a}} \cdot n _ {w}}{n _ {m} \cdot (1 - r) + n _ {w}} \backslash (6 \backslash)
$$

where  $n_m$  and  $n_w$  are the numbers of mainstream and We-media sources, respectively. This set of equations couples the population's collective emotional state  $(X_H, X_L)$  back to the information it receives  $(p_{\text{risk}})$ , forming a self-consistent system. By systematically varying the control parameter  $r$  from 0 (full mainstream media presence) to 1 (complete absence), we can study how the balance between negative and positive feedback shapes the system's macroscopic behavior.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/d70098c6cb7b570e029245430d6b91931757186e2aa36740bc3d3103f620973f.jpg)  
Figure 3 | Schematic of the Mixed-Feedback Model. The theoretical framework consists of three interacting components: the public (brown layer), mainstream media (yellow layer), and We-media (green layer). Mainstream media provides stabilizing negative feedback by counteracting the dominant public emotional state (left loop). In contrast, We-media provides amplifying positive feedback by reinforcing high-arousal states (right loop). The competition between these opposing mechanisms governs the system's collective emotional dynamics.

# Theoretical Analysis: Phase Transitions and Critical Phenomena in the Model

Having defined the model, we now systematically explore its emergent collective behaviors. We analyze how the system responds as we vary the external control parameter,  $r$ , across the internal parameter space defined by the psychological thresholds,  $\phi$  and  $\theta$ .

# Psychological Sensitivity as the Driver of System Fragility

Our analysis reveals that the system's stability is fundamentally governed by the psychological sensitivity of the population, represented by the interval  $\theta - \phi$ . As this gap narrows (i.e., sensitivity increases), the system becomes progressively more fragile and susceptible to abrupt transitions. This is illustrated in Figure 4, which shows how the system's emotional configuration changes in response to the removal of mainstream media ( $r$ ) for different values of the low-arousal threshold  $\phi$  (with  $\theta$  held constant).

As  $\phi$  increases, the moderate emotional state,  $X_{M}$ , is systematically compressed (Figure 4b,c). This erosion of the "emotional buffer" is the primary mechanism for instability. With fewer individuals in the moderate state to absorb fluctuations, the population is more easily pushed toward the extreme states of high or low arousal. This heightened fragility is quantitatively captured by both emotional responsiveness (Eq.12) and the jump amplitude (Eq.13) at the transition point, which grows systematically with  $\phi$  (Figure 4d), indicating more dramatic and abrupt shifts in the collective emotional state. For  $\phi > 0.41$ , the system's response becomes discontinuous; it first exhibits a high degree of "stickiness" or resistance to change, only to collapse suddenly when a critical threshold of  $r$  is crossed. This behavior highlights how psychological sensitivity and the balance of media feedback interact to determine whether the system responds to external pressures in a smooth, resilient manner or a brittle, discontinuous one.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/480e49eac35a21daf7eeda718e891958562f21d98dc34c90aaf8c66f3e564b3d.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/43fbaf869213bc7ebf7ea2cfbd94ed0f3ff1bd413902bf3c7a4f198ced0b8f5d.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/b5292527623c90f317fca1ea47bff2bb2d36215c73976e44935242a66d467d0a.jpg)  
Figure 4 | Psychological Thresholds Drive System Fragility. All panels show results from the mean-field model with a fixed high-arousal threshold  $\theta = 0.49$ . a, Emotional responsiveness, quantified as the rate of change of the high-arousal state with respect to the control parameter  $r$ , increases with the low-arousal threshold  $\phi$ . b,c, The evolution of the three emotional state fractions as  $r$  increases, for different values of  $\phi$ . As  $\phi$  increases toward  $\theta$ , the moderate state ( $X_M$ ) is compressed. d, The amplitude of the discontinuous jump in the high-arousal state at the transition point intensifies as  $\phi$  increases.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/88b3ec1fdd842e612f2f80909b1f9c4efb2070cb73878d77732f00b4c2c9659d.jpg)

# The Phase Diagram and Critical Boundaries

To map the full spectrum of the model's behavior, we constructed a phase diagram in the  $(\phi ,\theta)$  parameter space (Figure 5a). This diagram reveals three distinct regimes. A large region (dark teal) corresponds to a stable regime where the system's response to the control parameter  $r$  is always smooth and continuous. However, two critical regions emerge as the psychological sensitivity increases (i.e., as  $\phi$  approaches  $\theta$ ). A region shown in blue indicates the presence of first-order phase transitions, characterized by discontinuous, abrupt jumps in the collective emotional state. It is this specific dynamic regime of apparent stability followed by sudden, catastrophic collapse, predicted by our model, that we formally define as an "emotional bubble." A thin boundary region, shown in purple, marks the locus of second-order phase transitions, characterized by continuous but critical (i.e., non-analytic) changes.

This second-order boundary is of particular importance, as it serves as a critical frontier separating the stable and fragile regimes of the information ecosystem. As the system approaches this boundary, it becomes maximally sensitive to perturbations. The inset in Figure 5b provides a clear illustration of this phenomenon. At a fixed  $\theta = 0.49$ , as  $\phi$  is increased, the system's behavior transitions from a smooth, continuous response ( $\phi < 0.41$ ) to a critical second-order transition ( $\phi = 0.41$ ), and finally to a discontinuous, first-order jump ( $\phi > 0.41$ ). This demonstrates that the psychological parameters of the population are the ultimate determinant of the system's qualitative behavior, dictating whether it can gracefully adapt to change or is prone to sudden collapse.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/9a1a6ef7d7ec8355fac0fadd56f2ed5897de10827f0755daa075cb9bddddaef3.jpg)  
Figure 5 | Phase Diagram of Collective Emotional Dynamics. a, The phase diagram in the  $(\phi ,\theta)$  parameter space, mapping regions of stability (no phase transition, dark teal), first-order transitions (discontinuous, blue), and second-order transitions (continuous and critical, purple stripe). b, A magnified view for a fixed  $\theta = 0.49$ , showing the evolution of the high-arousal state  $X_{H}$  as a function of the control parameter  $r$ . As the low-arousal threshold  $\phi$  increases, the system's response shifts from continuous to a critical second-order transition and then to discontinuous first-order transitions.

# Hallmarks of Critical Phenomena

We confirmed the nature of these phase transitions by analyzing standard metrics of critical phenomena (Figure 6). At the second-order critical point (e.g.,  $\phi = 0.41, \theta = 0.49$ ), the system's correlation length,  $\xi$ , diverges at the critical value of the control parameter,  $r_c$  (Figure 6a). This divergence signifies that the system is developing long-range correlations, where small, local perturbations can have system-wide effects. Furthermore, this divergence follows a power law,  $\xi \sim \vee r - r_c i^{-\nu}$ , with a calculated critical exponent  $\nu \approx 0.94$  (Figure 6b).

The divergence and power-law scaling are defining hallmarks of a second-order phase transition.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/f5d3a7f9fb47f8918259c084b3bc6264eb43b03bb795eab0198f43e3dc571e94.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/ddf41cab42a0c3fc76e9f34c2a32c7e3ce991eeaae9e04c0c09caee6c87919c9.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/b80e15b438a4eb8a043fb9180fc611be786c8acbda59a597260942479e69a0b2.jpg)  
Figure 6 | Critical Phenomena at the Phase Transitions. a, At the second-order critical point  $(\phi = 0.41)$ , the correlation length  $\xi$  diverges at the critical removal ratio  $r_c$ . b, The power-law scaling of  $\xi$  near  $r_c$  on a log-log plot confirms the nature of the second-order transition, yielding a critical exponent  $\nu \approx 0.94$ . c, Emotional concentration shows a continuous change for the second-order transition but a discontinuous jump for first-order transitions (e.g.,  $\phi = 0.43$ ). d, The compression of the moderate state fraction,  $X_M$ , is most severe near the transition points, explaining the loss of system stability.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/c0041edbe8921bb8fbd48775139656385babf34193914aede430320014cb4723.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/263cd7bd76c3ead87606ad01f610f30d14c6e8e44c9b0adba013cf0107cd8d4f.jpg)

In contrast, the first-order transitions exhibit a discontinuous jump in the system's state variables. This is clearly visible in the emotional concentration,  $C$ , which shows an abrupt leap at the transition point for parameter values inside the first-order regime (Figure 6c). The underlying mechanism for both types of transitions is the compression of the moderate emotional state  $(X_M)$ , which acts as the system's stabilizing buffer. As shown in Figure 6d, the fraction of the population in the moderate state dips sharply near the transition points, with the effect being most pronounced as the system becomes more sensitive. This erosion of the emotional buffer is what renders the system brittle and susceptible to sudden, large-scale reorganization.

# Agent-Based Simulations: Validating the Mean-Field Approach and Exploring Microscopic Dynamics

To test the robustness of our mean-field theoretical predictions and to explore the individual-level dynamics, we implemented agent-based simulations on a network. In this framework, each individual is a node that makes state-transition decisions based on the information it receives from its local neighborhood of media sources.

This approach serves two critical purposes. First, it allows us to validate the mean-field approximation. We found excellent agreement between the macroscopic emotional state fractions predicted by the mean-field theory and those emerging from the agent-based simulations across the parameter space. The low Root Mean Square Error (RMSE) between the two approaches confirms the high fidelity of our theoretical calculations (Figure 7b,c). This agreement is a valuable finding in itself, demonstrating that the collective behavior of this system is not strongly dependent on the complex details of the network structure.

Second, the simulations provide a window into the microscopic transition dynamics that are averaged out in the mean-field approach. We measured the rate of direct transitions between the extreme emotional states (see Eq. 15), bypassing the moderate buffer state. This metric quantifies the system's underlying volatility. Our analysis revealed that these extreme-to-extreme transitions increase monotonically as psychological sensitivity  $(\phi)$  rises (Figure 7e). Furthermore, the rate of these transitions exhibits a U-shaped relationship with the media removal ratio,  $r$  (Figure 7f). Initially, a diverse media landscape  $(r \approx 0)$  creates a noisy environment with frequent transitions. As mainstream media is removed, the system becomes more homogeneous, allowing polarized opinions to stabilize and thus reducing transitions. However, beyond a certain point  $(r \gtrsim 0.8)$ , the near-total absence of the stabilizing negative feedback renders the system highly unstable, where small fluctuations can trigger large cascades, causing the transition rate to increase again. This non-monotonic behavior provides a microscopic explanation for why systems with either a highly diverse or a highly homogeneous media environment can be prone to instability.

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/cab307296a7562e9b58f29db8d51649bc0d1d9e966d8f8e45f52665acf91ef0f.jpg)  
a

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/72c18d7d1525b2c7c78fe0d0e89a4e640b2fa6960aac52feb40cb940408eadd4.jpg)  
b  
e

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/11c5184783c96546794fe9bab98878d72f683295359e9bb57317571b5eb4eb46.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/939f6c1cbb645529803b2d4e0ec3b363cf0e6078ba2e574427b76898337785fd.jpg)  
d

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/a797990b2728e0e9a442394d4ecdec6f4cd9e084307b8e579892c4a8a87be38e.jpg)  
Figure 7 | Network Simulation Analysis. a, A 3D plot from simulations showing the polarization level across the parameter space, consistent with mean-field predictions. b,c, Validation of the mean-field theory against agent-based simulations. The low RMSE between the theoretical and simulated steady-state proportions of  $X_{H}$  and  $X_{L}$  confirms the accuracy of the mean-field approach. d, Polarization metrics (Eq. 14) from simulations as a function of the control parameter  $r$ . e,f, Microscopic dynamics showing the rate of direct transitions between extreme emotional states ( $X_{H}$  and  $X_{L}$ ). The transition rate increases with psychological sensitivity  $\phi$  (e) and exhibits a U-shaped relationship with the media removal ratio  $r$  (f).

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-28/94cefba4-29fa-4183-aba1-0e5f2b089cd8/c16ebb2b0951bc40a47b52fe3ec8b1ee51917627a964625313d3a66178a0558c.jpg)  
f

# Discussion

This study introduced a novel theoretical framework, grounded in statistical physics, to investigate how the competition between stabilizing and amplifying feedback mechanisms in an information ecosystem can give rise to abrupt, nonlinear shifts in collective emotion. Motivated by the puzzling observation of opposing media-sentiment dynamics during the COVID-19 pandemic, we moved beyond empirical description to theoretical exploration. Our central contribution is not an empirical model of a specific crisis, but rather a minimal, mechanism-based model that reveals the fundamental conditions under which a social system can transition from a resilient state to a fragile one.

Our mean-field analysis and agent-based simulations yielded three core theoretical findings. First, the model demonstrates that the interplay between psychological sensitivity ( $\theta - \phi$ ) and the balance of media feedback ( $r$ ) is sufficient to produce a rich spectrum of collective behaviors, from smooth, continuous changes to sudden, discontinuous collapses. Second, we mapped the model's complete behavioral repertoire in a phase diagram, identifying distinct regions of stability, as well as the precise boundaries of first-order (discontinuous) and second-order (continuous, critical) phase transitions. Third, our analysis of critical phenomena, including the divergence of correlation length and power-law scaling ( $\nu \approx 0.94$ ), provides

quantitative, testable predictions about the nature of these transitions, offering theoretical early warning signals of systemic instability.

This theoretical framework provides a plausible, mechanism-based interpretation for the empirical puzzle that motivated our inquiry. The inverted U-shaped relationship observed between mainstream media exposure and emotional polarization is consistent with the behavior of the stabilizing negative feedback mechanism in our model, which can moderate polarization under certain conditions. Conversely, the linear relationship seen with We-media aligns with the amplifying positive feedback loop, which consistently drives the system toward more extreme states. Most importantly, the model's prediction of first-order phase transitions—the "emotional bubbles"—offers a compelling theoretical explanation for how a seemingly stable public mood can suddenly and catastrophically collapse, a phenomenon that linear models cannot capture. Our model suggests that such fragility is an emergent property of the system, arising from the compression of the moderate emotional buffer as public sensitivity to risk information increases. Furthermore, our agent-based simulations offer a potential microscopic explanation for another of our empirical findings. The U-shaped relationship observed between total risk exposure and emotional concentration (Figure 2b) is strikingly mirrored by the U-shaped relationship between the media removal ratio and the rate of direct transitions between extreme emotional states in our simulations (Figure 7f). This suggests that macroscopic emotional clustering may be mechanistically linked to the underlying volatility at the individual level, where a lack of moderating feedback—either in a noisy or a heavily biased information environment—facilitates direct, polarizing jumps in sentiment.

It is crucial to acknowledge the limitations of our minimalist theoretical model, which point toward important avenues for future research. First, our model simplifies the information ecosystem by treating all media sources within a category as unweighted. Future work should incorporate greater realism by weighting media influence by factors such as follower count, credibility, or engagement metrics. Second, our model assumes a homogeneous population. Decomposing the population into demographic groups with different psychological thresholds, informed by empirical studies on sociodemographic differences in emotional responses  $^{27,28}$ , would be a valuable extension.

Third, our agent-based simulations, while useful for validating the mean-field approach, were conducted on a random network. Integrating more realistic social network topologies, which are known to exhibit features like community structure and homophily  $^{29-31}$ , could reveal how network structure modulates the collective emotional dynamics. Finally, and most importantly, while this work was motivated by data, the model itself is purely theoretical. A crucial next step would be to conduct a rigorous out-of-sample validation by fitting the model's parameters to data from one crisis (e.g., the first wave of COVID-19) and testing its ability to predict the emotional dynamics of a subsequent, different event (e.g., a later wave or a different type of public health crisis). Such an exercise would represent a true test of the model's predictive power.

In conclusion, our study demonstrates that the competition between opposing feedback mechanisms is a powerful driver of systemic fragility in collective emotional systems. By

viewing these dynamics through the lens of phase transition theory, we can move beyond simple descriptions of what happened during a crisis, and begin to build a deeper, mechanism-based understanding of why it happened. This theoretical foundation is essential for developing more effective and adaptive strategies to foster information ecosystem resilience and safeguard public mental health during times of profound uncertainty.

# Materials and Methods

This section provides the technical details for the empirical analysis, mean-field theoretical framework, and computational simulations presented in the main text.

# Empirical Data and Processing

The empirical analysis was based on three datasets collected from Weibo from January 2020 to December 2023: 49,288 posts from verified We-media accounts, 36,480 posts from verified mainstream media outlets, and 208,666 posts from individual users. Text preprocessing involved standard procedures, including the removal of special characters, emojis, and short texts (<10 characters).

For the machine learning classification, 5,000 posts were randomly sampled and manually annotated for emotional arousal (high, medium, low) and risk perception (risk, no-risk) by three trained volunteers. After quality control, 4,086 labeled samples were used to fine-tune a DistilBERT-based model for multi-task classification  $^{32}$ . The model's robustness was validated through 100 independent experiments with different train-test splits. The final high-performing classifiers were used to label the entire corpus, forming the basis of the descriptive statistics presented in the "Empirical Motivation" section. Further details on the classification keywords and model performance are provided in the Supplementary Information.

# Empirical Pattern Analysis

To quantify the nonlinear relationships between media exposure and collective emotional responses, we developed two key metrics.

Emotional Concentration. We measure emotional clustering using Shannon entropy:

$$
S = - \sum_ {i = 1} ^ {3} \square p _ {i} \log_ {2} p _ {i} (7)
$$

where  $p_i$  represents the relative frequency of emotional state  $i$  (low, medium, high arousal) in daily observations. The emotional concentration index is then:

$$
C = 1 - \frac {S}{S _ {\text {m a x}}} (8)
$$

where  $S_{max} = \log_2 3$  is the maximum possible entropy for three emotional states.

Polarization Index. We quantify emotional polarization as the ratio of extreme to moderate emotional states:

$$
P = \frac {p _ {\text {h i g h}} + p _ {\text {l o w}}}{p _ {\text {m i d d l e}}} (9)
$$

These metrics capture the key empirical patterns that motivated our theoretical framework development.

# Mean-Field Theoretical Modeling

The theoretical analysis is based on the self-consistent equations (Eqs. 1-6) described in the main text. The system's equilibrium state for a given set of parameters  $(r, \phi, \theta)$  was found by iteratively solving these coupled equations until the population fractions  $(X_H, X_M, X_L)$  converged to a steady state with a tolerance of  $\epsilon < 10^{-6}$ .

To analyze the stability of these steady-state solutions and identify phase transitions, we employed Jacobian matrix analysis. The Jacobian matrix  $J$  of the self-consistent system was computed via numerical differentiation. The stability of a solution is determined by the eigenvalues,  $\lambda$ , of this matrix. A solution is stable if all eigenvalues have negative real parts. A phase transition occurs when the real part of the largest eigenvalue,  $\lambda_{\max}$ , crosses zero.

The correlation length,  $\xi$ , a key metric for characterizing phase transitions, was calculated as the inverse of the absolute value of the largest eigenvalue's real part:

$$
\xi = \frac {1}{\dot {\iota} \operatorname {R e} (\lambda_ {m a x}) \vee \dot {\iota} \backslash (1 0 \backslash) \dot {\iota}}
$$

Near a second-order phase transition, this correlation length is expected to diverge following a power law:

$$
\xi \sim \vee r - r _ {c} i ^ {- v} \backslash (1 1 \backslash)
$$

where  $r_c$  is the critical value of the control parameter and  $\nu$  is the critical exponent. We determined  $r_c$  by identifying the peak of  $\xi$  and calculated  $\nu$  by performing a linear fit on a log-log plot of  $\xi$  versus  $\dot{\iota} r - r_c \vee \dot{\iota}$ .

To analyze system fragility and transition dynamics, we computed two additional metrics:

Emotional Responsiveness. We quantify the system's sensitivity to media composition changes as:

$$
R = \left| \frac {d X _ {H}}{d r} \right| (1 2)
$$

where  $X_H$  is the high-arousal population fraction and  $r$  is the mainstream media removal ratio.

Jump Amplitude. At critical transition points, we measure discontinuous shifts as:

$$
J = \max  \left(X _ {H} ^ {\text {w i n d o w}}\right) - \min  \left(X _ {H} ^ {\text {w i n d o w}}\right) (1 3)
$$

where  $X_H^{\mathrm{window}}$  represents high-arousal proportions within a parameter window around transition points. The window size is 5.

# Agent-Based Network Simulations

The agent-based simulations were implemented to validate the mean-field approach and explore microscopic dynamics. The network consisted of three types of nodes:  $n_m = 50$  mainstream media sources,  $n_w = 50$  We-media sources, and  $n_p = 10,000$  public individuals. Information flow was modeled as directed edges from media nodes to public nodes. Each public node's in-degree was drawn from a Poisson distribution with a mean  $\langle k \rangle = 10$ .

At each time step of the simulation, the following updates occurred: (1) Each public node  $i$  determined its new emotional state  $(H, M, \text{or } L)$  based on the fraction of risk signals received from its  $k_i$  connected media sources, according to its individual thresholds,  $\phi_i$  and  $\theta_i$ . (2) After all public nodes were updated, each media node updated its risk perception based on the new average emotional state of the public nodes it connected to, following the feedback rules in Eqs. 4 and 5. The simulation was run until the system reached a macroscopic steady state, and results were averaged over multiple independent runs to ensure robustness. Microscopic transition rates were calculated by tracking the proportion of individuals switching directly between the  $H$  and  $L$  states within a given time window after the system had stabilized.

To quantify system polarization and microscopic transition dynamics, we computed:

Polarization Metric. We measure system polarization as the absence of moderate states:

$$
\text {P o l a r i z a t i o n} = 1 - X _ {M} (1 4)
$$

where  $X_{M}$  is the fraction of individuals in moderate arousal states.

Direct Transition Rate. We track the proportion of individuals switching directly between extreme states, bypassing the moderate buffer:

$$
T _ {H \leftrightarrow L} = T _ {H \rightarrow L} + T _ {L \rightarrow H} (1 5)
$$

where  $T_{ij}$  represents the probability of transitioning from state  $i$  to state  $j$  within a given time window. This metric quantifies the system's stability against extreme oscillations.

# Code and Data Availability

All code and data supporting the conclusions of this article are publicly available. The complete computational framework, including data preprocessing scripts, machine learning models, theoretical calculations, and network simulations, is accessible at https://github.com/XX/MFM. The repository contains detailed implementation of the mixed-feedback model (MFM), self-consistent equation solvers, phase transition analysis tools, and agent-based network simulation codes. Raw data from Weibo posts, processed datasets, and trained model parameters are available through the same repository. All analyses were conducted using Python 3.8 with standard scientific computing libraries (NumPy, SciPy, pandas, scikit-learn, NetworkX). The theoretical calculations utilize custom-developed numerical solvers for self-consistent equations and Jacobian matrix analysis. Computational requirements and detailed installation instructions are provided in the repository documentation to ensure full reproducibility of all results presented in this study.

# Funding Declaration

This study was funded by the National Social Science Foundation of China (Grant No. 24AXW005, “Research on Telling Chinese National Stories from the Perspective of Strengthening the Chinese National Community Consciousness”). The funder played no role in study design, data collection, analysis and the writing of this manuscript.

# Author Contribution Statement

LJ acquired the funding and resources, and provided supervision. JL designed the research framework, constructed the model, performed the formal analysis, and was a major contributor in writing the manuscript. ZH developed the software and code for the project, and contributed to writing the manuscript. All authors read and approved the final manuscript.

# Competing interests

The authors declare no competing interests.

# Reference

1. Centola, D. The Spread of Behavior in an Online Social Network Experiment. Science (2010) doi:10.1126/science.1185231.  
2. Watts, D. J. A simple model of global cascades on random networks. Proc. Natl. Acad. Sci. 99, 5766-5771 (2002).  
3. Becker, J., Brackbill, D. & Centola, D. Network dynamics of social influence in the wisdom of crowds. Proc. Natl. Acad. Sci. 114, E5070-E5076 (2017).  
4. Hasell, A. Shared Emotion: The Social Amplification of Partisan News on Twitter. Digit. Journal. (2021).  
5. Chong, M. & and Choy, M. The Social Amplification of Haze-Related Risks on the Internet. Health Commun. 33, 14–21 (2018).  
6. Lerman, K., Feldman, D., He, Z. & Rao, A. Affective polarization and dynamics of information spread in online networks. Npj Complex. 1, 8 (2024).  
7. Helfmann, L., Djurdjevac Conrad, N., Lorenz-Spreen, P. & Schütte, C. Modelling opinion dynamics under the impact of influencer and media strategies. Sci. Rep. 13, 19375 (2023).  
8. Giacomo, A., Calzola, E. & Dimarco, G. Opinion Dynamics in Social Networks: Kinetic and Data-driven Modeling | SIAM. Society for Industrial and Applied Mathematics https://www.siam.org/publications/siam-news/articles/opinion-dynamics-in-social-networks-kinetic-and-data-driven-modeling/ (2025).  
9. Piao, J., Liu, J., Zhang, F., Su, J. & Li, Y. Human-AI adaptive dynamics drives the emergence of information cocoons. Nat. Mach. Intell. 5, 1214-1224 (2023).  
10. Green, J., Edgerton, J., Naftel, D., Shoub, K. & Cranmer, S. J. Elusive consensus: Polarization in elite communication on the COVID-19 pandemic. Sci. Adv. 6, eabc2717 (2020).  
11. Bavel, J. J. V. et al. Using social and behavioural science to support COVID-19 pandemic response. Nat. Hum. Behav. 4, 460–471 (2020).  
12. Wang, J. et al. Global evidence of expressed sentiment alterations during the COVID-19 pandemic. Nat.

Hum. Behav. 6, 349-358 (2022).  
13. Aslam, F., Awan, T. M., Syed, J. H., Kashif, A. & Parveen, M. Sentiments and emotions evoked by news headlines of coronavirus disease (COVID-19) outbreak. Humanit. Soc. Sci. Commun. 7, 23 (2020).  
14. Chou, W.-Y. S. & and Budenz, A. Considering Emotion in COVID-19 Vaccine Communication: Addressing Vaccine Hesitancy and Fostering Vaccine Confidence. Health Commun. 35, 1718-1722 (2020).  
15. Liu, Z., Wu, J., Wu, C. Y. H. & Xia, X. Shifting sentiments: analyzing public reaction to COVID-19 containment policies in Wuhan and Shanghai through Weibo data. Humanit. Soc. Sci. Commun. 11, 1104 (2024).  
16. Kasperson, R. E. et al. The Social Amplification of Risk: A Conceptual Framework. *Risk Anal.* 8, 177-187 (1988).  
17. Jorgensen, F., Bor, A., Rasmussen, M. S., Lindholt, M. F. & Petersen, M. B. Pandemic fatigue fueled political discontent during the COVID-19 pandemic. Proc. Natl. Acad. Sci. 119, e2201266119 (2022).  
18. Goldenberg, A., Weisz, E., Sweeny, T. D., Cikara, M. & Gross, J. J. The Crowd-Emotion-Amplification Effect. Psychol. Sci. 32, 437–450 (2021).  
19. Brady, W. J. et al. Overperception of moral outrage in online social networks inflates beliefs about intergroup hostility. Nat. Hum. Behav. 7, 917-927 (2023).  
20. Crockett, M. J. Moral outrage in the digital age. Nat. Hum. Behav. 1, 769-771 (2017).  
21. Kramer, A. D. I., Guillory, J. E. & Hancock, J. T. Experimental evidence of massive-scale emotional contagion through social networks. Proc. Natl. Acad. Sci. U. S. A. 111, 8788-8790 (2014).  
22. Fan, R., Zhao, J., Chen, Y. & Xu, K. Anger Is More Influential than Joy: Sentiment Correlation in Weibo. PLOS ONE 9, e110184 (2014).  
23. Bakshy, E., Messing, S. & Adamic, L. A. Exposure to ideologically diverse news and opinion on Facebook. Science 348, 1130-1132 (2015).  
24. Wilson, O. & Flahault, A. China's U-turn in its COVID-19 policy. Anaesth. Crit. Care Pain Med. 42, 101197

(2023).  
25. Del Vicario, M. et al. The spreading of misinformation online. Proc. Natl. Acad. Sci. U. S. A. 113, 554-559 (2016).  
26. Del Vicario, M. et al. Echo Chambers: Emotional Contagion and Group Polarization on Facebook. Sci. Rep. 6, 37825 (2016).  
27. Schwartz, H. A. et al. Personality, Gender, and Age in the Language of Social Media: The Open-Vocabulary Approach. PLOS ONE 8, e73791 (2013).  
28. Pfefferbaum, B. & North, C. S. Mental Health and the Covid-19 Pandemic. N. Engl. J. Med. 383, 510–512 (2020).  
29. Bisgin, H., Agarwal, N. & Xu, X. A study of homophily on social media. World Wide Web 15, 213-232 (2012).  
30. Dandekar, P., Goel, A. & Lee, D. T. Biased assimilation, homophily, and the dynamics of polarization. Proc. Natl. Acad. Sci. 110, 5791-5796 (2013).  
31. McPherson, M., Smith-Lovin, L. & Cook, J. M. Birds of a Feather: Homophily in Social Networks. Annu. Rev. Sociol. 27, 415-444 (2001).  
32. Sanh, V., Debut, L., Chaumont, J. & Wolf, T. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. Preprint at https://doi.org/10.48550/arXiv.1910.01108 (2020).