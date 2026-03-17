# Complementary Information Flows Drive Emotional Dynamics in Long-COVID Discourse

# Abstract

Long-COVID discourse generates complex emotional patterns across information channels, yet the mechanisms driving these dynamics remain poorly understood. We introduce the Content-Sentiment Directed Approval Graph (CSDAG) to decode emotional state transitions in public health communication. Analyzing social media posts across mainstream media, self-media, and public users, we reveal a dual-pathway architecture governing emotional response: mainstream media provides structural stability by systematically moderating extreme emotional states, while We Media enables rapid emotional adaptation through responsive content dynamics. State transition analysis demonstrates these complementary pathways create a self-regulating system where high-arousal emotions are channeled toward moderate states ${ \cdot } p { < } 0 . 0 0 1 )$ , explaining previously observed emotional divergence between institutional and social media during health crises. Our model achieves robust predictive accuracy $\mathrm { R M S E } { = } 0 . 1 2 5$ ), significantly outperforming baseline models, and reveals optimal crisis communication emerges when these complementary information flows are balanced. These findings offer critical insights for managing public emotional dynamics during health emergencies by leveraging the distinct regulatory functions of different communication channels. 

Keywords: Multilayer networks, Agent-based modeling, Emotional transition, Dualchannel adaptation 

The COVID-19 pandemic has significantly impacted various aspects of human society, including economic development1, social justice2, and mental health3. While the international community has worked to mitigate these immediate effects, recent research has increasingly shifted focus to a distinct group of individuals, often referred to as “Long-haulers’’, who experience prolonged symptoms following COVID-19 infection4,5 Unlike the acute phase of COVID-19, which typically resolves within two to three weeks, Long-COVID involves symptoms lasting over 12 weeks. These symptoms include fatigue, loss of taste or smell, memory problems, concentration difficulties, and cardiovascular, without other medical explanations6–8. 

The definition and duration of Long-COVID remain subjects of debate among researchers9–11, leading to inconsistencies in public discourse and generating anxiety and frustration among those affected10,12. This emotional distress, combined with uncertainty about the disease’s future and insufficient medical care, heightens the risk of suicidality and negatively impacts individual well-being13. Before it became a focal point, patient communities on social media were already exchanging experiences and garnering support14,15. These patient-led, decentralized information channels have evolved into a multi-actor public sphere, where like-minded communities create and share health-related content independently of traditional authorities16. Given the importance of public mental health, assessing the impact of decentralized health communication on emotional shifts is vital for shaping prevention policies2. This study focuses on how fear appeals and different media sources shape emotional responses during the Long-COVID, and how the cognitive processing of information moderates these effects on public sentiment. 

Despite numerous studies on media’s influence on public sentiment17–20, critical gaps remain in understanding the dynamic interplay between media messages and audience feedback, particularly in networked, time-sensitive contexts22–28, and the function of like-minded communities on public emotional responses24. Current models relying on survey data or traditional statistical analyses often fail to capture these complex emotional transitions across multi-centered social platforms29-30. To address these gaps, we utilize agent-based modeling (ABM) combined with real network data to analyze the intricate relationships between public sentiment and media narratives within these dynamic environments. By simulating individual and group behaviors, ABM bridges micro-level interactions with broader societal trends, offering a comprehensive approach to understanding how decentralized health communication influences public sentiment31–34. To further this analysis, we propose the Content-Sentiment Directed Approval Graph (CSDAG), a heterogeneous network model that describes strategic behaviors and information flow among diverse stakeholders (see Fig. 1). The model categorizes stakeholders with different motives participate in the information diffusion process. This graph is segmented into three distinct set of nodes: 

the mainstream media, $M$ ; followed by the We Media,W . Both sets utilize content as the primary means of exerting social influence. In contrast, the ordinary people constitute the other node set S, where people sentiments are treated as response for media information. We identify these stakeholders by analyzing keywords related to industry, workplace, and profile descriptions (Supplementary Section 6). To explore the interrelationships among different stakeholders, we propose several critical assumptions as follows: 

As shown in Fig. 1, the upper layer represents the mainstream media-driven sphere. In China’s public life, the primary mode of information flow and diffusion remains topdown, driven by the government as the principal information source29,30. These outlets are highly credible and authoritative, reaching a wide audience. They are crucial in shaping the public agenda31, disseminating information32, and influencing opinions and behaviors33,34. Mainstream media are often viewed as government extensions, necessitating a focus on public interest and political stance in their operations35. When public discussions revolve around Long-COVID, mainstream media need to guide opinions. Barbieri et al. argue that media establish authority through content production36. On social media, mainstream outlets extend their influence by amplifying fear appeals37 related to Long-COVID through microblogging, significantly shaping public understanding and emotional responses38. Fear appeals can trigger a sequence of emotions in viewers, with emotional states evolving as the message unfolds39,40. Compared to positive sentiments, negative emotions, such as anger and fear, draw more attention41 and are particularly powerful in driving social mobilization17,42. Negative framing information directly influences the perceptions of ordinary people, as evidenced by COVID-19 coverage that emphasizes infection counts and mortality over recovery rates, which heightens negative emotions and sensitizes individuals to perceived risks43. 

Unlike mainstream media, We Media adopts a decentralized and customized approach, quickly adapting to and reflecting audience interests to attract more attention and engagement (see Fig. 1). We Media, driven by profit maximization, tailors content to individual users’ psychological tendencies and preferences44, enhancing engagement and increasing advertising revenue45. It operates on its logic, shaped by programmability, popularity, connectivity, and datafication, which together drive content circulation46. On the other hand, Chinese mainstream media still set agendas and define political boundaries47. To gain legitimacy, We Media must balance aligning with mainstream media while also creating content that resonates with its audience, fostering connectivity and customizing content to reflect audience preferences. 

In the bottom layer of Fig. 1, ordinary people, positioned at the end of the information chain, are influenced by media content, with their emotional responses feeding back to shape subsequent media production. People receive Long-COVIDrelated information through news feeds, which elicits emotional and behavioral reactions48; fear appeals, for instance, can evoke emotions like anger, disgust, or sadness49. Emotional arousal levels directly impact how individuals perceive and respond to information1,50,51. High-arousal states, such as anger or fear, increase cognitive load and focus52, making people more receptive to structured, authoritative 

information. Conversely, moderate arousal reduces cognitive load, allowing greater openness to diverse and complex information53. Low-arousal states like boredom can diminish engagement54 and distort perception55, while moderate anxiety may enhance navigation through complex informational environments56,57. Additionally, individuals’ emotional states are influenced by like-minded peers,58 reinforcing homophily as people preferentially engage with those they perceive as similar59,60. This tendency strengthens the social feedback loop61, where emotions affect media content and dissemination patterns62. The government and mainstream media prioritize maintaining social order63, adjusting content to align with public sentiment. Meanwhile, We Media adapts to sentiment feedback to sustain popularity within its user base. 

In this study, we investigate how information flows across media channels shape public emotional responses to Long-COVID. Analyzing Weibo data during China's transition from zero-COVID policy, we develop the Content-Sentiment Directed Approval Graph (CSDAG) to decode emotional state transitions in networked communication environments. Our findings reveal a sophisticated dual-pathway regulatory system governing public emotional dynamics. Mainstream media functions as an emotional intensification mechanism through an “authority amplification effect.” When official sources report risks, their heightened credibility concentrates public attention and often contradicts expectations of stability, amplifying emotional responses toward higher arousal states. Conversely, We Media operates through an “attention diffusion mechanism” where fragmented, diverse content creates multidirectional emotional effects—simultaneously dampening certain transitions while moderately amplifying others as competing narratives distribute attention and counterbalance each other. Most significantly, these pathways interact in unexpected ways. Rather than experiencing amplified emotional responses, users connected to both media types demonstrate enhanced emotional regulation capabilities. For these individuals, risk information through either channel significantly increases transitions from high to middle arousal states—revealing a counterintuitive buffering effect that challenges assumptions about cumulative exposure to risk information. This complementary system achieves robust predictive accuracy $\mathrm { R M S E } { = } 0 . 1 2 5$ ) and illuminates effective crisis communication strategies. Optimal emotional management during health emergencies requires balancing authoritative guidance with diverse information channels—mainstream media provides necessary information that mobilizes appropriate concern, while We Media offers contextualizing perspectives that modulate extreme emotional responses. This dual-channel approach creates a self-regulating system capable of both activating and stabilizing public emotional responses, offering crucial insights for managing ongoing and future health emergencies where both emotional fatigue and insufficient risk awareness present significant challenges. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/4714bd446384b07f4396b238403131b52070ee38ef6cc9da63626dfecd23e64b.jpg)



Figure 1. Content-Sentiment Directed Approval Graph (CSDAG). The yellow layer represents state-controlled media, the green layer depicts We Media ecology, and the red layer signifies the ordinary people’s network. Solid arrows illustrate the pathways of influence, while dashed lines denote the connections between media entities and individuals.


# Results

As mentioned, we aim to refine the analytical framework for health information dissemination using ABM. This model integrates various agents—ordinary people, mainstream media, and We Media—each defined by attributes derived from empirical data. These agents are connected by forwarding acts, which represent approval and facilitate the flow of information. Our ABM aims to uncover the interplay between public emotions and media content by simulating the decision-making processes of these agents based on the information they have received. This approach promises to deepen our understanding of the dynamic relationship between media content related to health risks and public emotional responses. We conducted simulations on the proposed network, CSDAG, along with the empirical data calibration. These experiments enable 

us to observe how the theoretical rules bring about the outcome, revealing the Longitudinal causal mechanisms and vividly validating the proposed hypothesis64,65. 

We calibrate vital parameters, including the parameter controlling the state changes, to fit the actual statistics obtained from empirical data. ABM experiments begin on October 14th, 2022, and end on January 18th, 2023. 

# Empirical validation of emotional dynamics through media channels

Figure 2 reveals distinct emotional regulation mechanisms across CSDAG layers. The mainstream media layer demonstrates structured content production that systematically modulates public emotional states, primarily stabilizing middle-arousal responses through authoritative information dissemination. The We Media layer exhibits more dynamic content adaptation, rapidly responding to and amplifying public emotional signals, particularly in high-arousal states. The public layer shows rich emotional transition patterns, with state stability enhanced by group emotional homophily. Crosslayer analysis reveals bidirectional feedback: while media content shapes initial emotional responses, public sentiment reciprocally influences subsequent content production, particularly in We Media channels. This differentiated regulation mechanism explains how public emotional dynamics emerge from the interplay between structured guidance and adaptive feedback. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/589e6be128a7da2321a037a5398e6fbfa378004097756c08eafdf0bed55ced68.jpg)



Figure 2. Content-driven emotional regulation in health crisis communication. The Mainstream Media layer (top) demonstrates structured content regulation, with risk/non-risk signals (red/blue) providing emotional anchoring. The We Media layer (middle) shows adaptive content production responding to public sentiment. The Public layer (bottom) displays emotional state transitions (high/middle/low arousal, shown in dark blue/green/light orange). Arrows indicate content-emotion interaction pathways and feedback mechanisms.


To rigorously examine the causal relationships between media content and emotional dynamics, we conducted comprehensive Granger causality tests across multiple lag structures (See supplementary section 4). The analysis revealed distinct temporal influence patterns in the information-emotion network. Mainstream media demonstrated significant immediate effects on emotional states, particularly strong for middle-arousal emotions $\scriptstyle ( \chi ^ { 2 } = 1 1 . 3 3 1$ , $p { = } 0 . 0 0 1$ at lag 1) and high-arousal emotions ( $\chi ^ { 2 } { = } 5 . 1 7 5$ , $\scriptstyle { p = 0 . 0 2 3 }$ at lag 1). The influence on low-arousal emotions emerged more gradually, becoming highly significant at longer time scales $\scriptstyle ( \chi ^ { 2 } = 2 9 . 0 8 4$ , $p { < } 0 . 0 0 1$ at lag 3). 

We Media showed a more focused pattern of influence, with strongest immediate 

impact on high-arousal emotions $( \chi ^ { 2 } = 5 . 6 0 3$ , $\scriptstyle { p = 0 . 0 1 8 }$ at lag 1), while its effect on middle-arousal emotions became significant only at longer time scales $( \chi ^ { 2 } = 8 . 0 9 5$ , $\scriptstyle { p = 0 . 0 4 4 }$ at lag 3). This temporal structure suggests that We Media primarily drives rapid emotional responses, while mainstream media maintains more sustained influence across emotional states. 

The analysis also revealed significant feedback effects: high-arousal emotions showed strong influence on low-arousal states $( \chi ^ { 2 } = 4 . 8 1 3$ , $\scriptstyle { p = 0 . 0 2 8 }$ at lag 1), while both high and middle-arousal states demonstrated significant longer-term impacts on mainstream media content $\scriptstyle ( \chi ^ { 2 } = 2 1 . 0 2 5$ and $\chi ^ { 2 } { = } 3 2 . 6 3 7$ respectively, $p { < } 0 . 0 0 1$ at lag 3). These bidirectional relationships highlight the complex dynamics between media content and emotional responses in crisis communication. 

This hierarchical influence structure aligns with Bail et al.’s findings on mediadriven emotional cascades in digital networks66. At the same time, the differentiated impact patterns support Goldenberg and Gross’s framework of emotion-specific transmission mechanisms67. The temporal sequence and effect magnitudes demonstrate that institutional media channels maintain primary influence over emotional responses. At the same time, We Media serves as a secondary, yet significant, driver of higharousal emotional states. 


a


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/1ad089a52a5e4f49faa8bd8c62e29493e09a24851e3a3e95886dc77d2e82b37f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/3e98da6039fcbe7d1fcc9a11601e39389b331281b43773d05e8c54db38abd02f.jpg)



Edge width represents -log10(p-value)



Figure 3. Granger causality analysis reveals distinct temporal patterns in Long-COVID discourse networks. a, Time series of emotional proportions and media risk reporting from September 2019 to December 2020. High (red), middle (yellow), and low (blue) arousal emotional states are shown alongside mainstream media (green) and We Media (orange) risk reporting patterns. b, Network visualization of significant Granger causal relationships between media channels and emotional states. Node sizes represent in-degree centrality, and edge widths indicate - log10(p-value) of the Granger causality test.


# Model Validation and Dual-Channel Information Flow Dynamics

The CSDAG model demonstrates robust convergence and predictive capability across multiple dimensions (Fig. 4). The parameter optimization process steadily improves, 

with mean error decreasing from 0.165 to 0.129 over ten iterations, while the minimum error stabilizes at $R M S E { = } 0 . 1 2 5$ (Fig. 4a). The time series validation (Fig. 4b) reveals powerful performance in tracking high-arousal emotional transitions and We Media risk reporting patterns, capturing both gradual trends and sudden fluctuations. 

The correlation structure (Fig. 4d) uncovers a nuanced hierarchy in media-emotion coupling. We Media exhibits strong positive correlation with high-arousal emotions (0.76) while negatively correlating with low-arousal states (-0.85), suggesting its role in emotional amplification. Mainstream media demonstrates an inverse pattern: moderate positive correlation with high-arousal emotions (0.48) but stronger negative correlation with middle-arousal states (-0.64), indicating its stabilizing influence on emotional dynamics. 

The mutual information analysis (Fig. 4c) further quantifies this asymmetric influence, showing highest prediction fidelity for high-arousal emotions (0.80) and We Media risk reporting (0.76), in contrast to mainstream media's lower mutual information (0.39). This dual-channel mechanism demonstrates how institutional risk messaging and social feedback complement each other in modern crisis communication. This asymmetric predictive performance aligns with the complex contagion framework established through controlled social media experiments68, where information adoption depends critically on the diversity of sources rather than mere exposure frequency. The distinct correlation patterns reveal how different media channels shape emotional dynamics: We Media, representing diverse information sources, drives emotional amplification through rapid feedback loops, while mainstream media exerts a more measured, regulatory influence. This dual-channel mechanism echoes recent findings that information spreading in online social systems follows complex threshold dynamics69, where multiple distinct sources are necessary for effective emotional contagion. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/b1c3d070a5863b68d655b2b82b0a982a7414f33c95eb3c94fe9b12997aa279d0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/e6a11ec0aabfac9739a89243868c5c83a1804c7f484878f4728a36b49bf9f98d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/a8897e93420675aba6999720d2b69a6b23238a0321cc949b80c0db58cdf79731.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/7528a16a96bcc7e654559299a42b742fb1160794cff016e67accc76aeb9322c3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/e42a240844a325e80cba3f3b8d3254301906952fa8b49de5b316f370e019a0e0.jpg)



Figure 4. Model validation and information flow analysis in the CSDAG framework. a, Convergence of mean error (green line) and minimum error (red line) across ten iterations of parameter optimization. b, Time series comparison between simulated (lines) and empirical (points) data for emotional states and media risk reporting. c, Quantitative assessment of model performance through mutual information (top) and sliding window RMSE (bottom) between simulated and empirical time series. d, Correlation matrix revealing the coupling strength between emotional states and media risk reporting, with color intensity indicating correlation magnitude from -1 (purple) to 1 (navy).


The parameter distributions in our CSDAG models reveal fundamental mechanisms underlying media-emotion dynamics. The We Media sensitivity parameter $\beta$ exhibits the highest sensitivity (0.869) with a concentrated distribution (0.4-1.2), suggesting a consistent yet bounded influence pattern in emotional regulation. This contrasts with mainstream media sensitivity $\alpha$ , which shows notable instability across iterations, indicating its effects may be highly context-dependent and mediated by user connection patterns. 

The homophily parameter ζ demonstrates robust sensitivity (0.683) with convergent distributions (0.2-0.6), revealing the fundamental role of community-level social reinforcement in emotional dynamics69. Together with the moderate sensitivities of fear appeal effect θ (0.222) and regulatory field strength γ (0.291), these patterns point to a dual-channel regulatory system: while We Media maintains stable adaptive responses through distributed networks, mainstream media exerts variable influences through institutional pathways, with their combined effects modulated by community structures. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/63339e74aff92d8979d222cf090ab8be0e9dfff985dc76ee6eb06d815242b310.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/7e860a005c5863f4c7da155840b3ab60e5654925c9abe3a9c8033766e2530e74.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/6eaafb923e10f25484feda880d470f5a04fd9c2c1acdda06a8dca18dbfeb6f73.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/1cc09847f636bf9410e362c183eb884fe30f2ffd837d36094ba38d0b9d7be4c3.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/f30ffaa82ee19bff39609d849368705965796068fc837ac104755df22d8083eb.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/55f66590d38ef41ec16545d05bdafaf39c691af725af786701518d7274dfa7ea.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/11a33ccfcf8f548d0c73e6022c43c239230d9a5f97e90b146634de8e17f869a2.jpg)



Figure 5. Parameter distributions in the CSDAG model across Approximate Bayesian Computation iterations. The ridge plots show the posterior distributions of seven key parameters: mainstream media sensitivity $( \alpha )$ , We Media adaptability $( \beta )$ , media fear appeal effect (θ), social influence threshold $( \sigma )$ , homophily-driven community influence $( \zeta )$ , adaptive regulatory field strength (γ), and influence acceptance rate $( \mu )$ . Colors transition from purple (early iterations) to yellow (final iterations), representing the evolution of parameter distributions across ten iterations.


# Dual-Channel Mechanisms of Media Risk Reporting in Emotional State Transitions

The discovery of contrasting correlations between moderate arousal emotions and risk reporting (mainstream media: -0.64, We media: 0.13; Fig. 4d) led us to examine emotional transitions through the lens of moderate arousal as a critical threshold state. We decomposed emotional dynamics into four fundamental transition pathways (see method) : intensification (upward), moderation (downward), stabilization (horizontal), and activation (low-level rise). Our state transition probability analysis reveals precise mechanisms through which media risk reporting influences emotional dynamics, transcending limitations of traditional correlation studies (Fig. 6a). The data demonstrates distinct regulatory effects of mainstream and We Media risk reporting on emotional transitions. Mainstream media risk reporting significantly increases the probability of transitions from medium to high arousal states $M  H$ increased by $1 7 . 5 5 \%$ , $p { < } 0 . 0 0 1 $ ) and promotes activation of low arousal states ( $L \to M$ increased by $1 . 0 8 \%$ , $p { < } 0 . 0 0 1 $ , forming a clear upward emotional gradient. In contrast, We Media risk reporting’s impact on emotional intensification is only one-quarter that of mainstream media ( $M \to H$ increased by $4 . 8 4 \%$ , $p { = } 0 . 0 3 $ ), and exhibits an inhibitory effect on emotional activation ${ \bf \cal L } \to { \cal M }$ decreased by $0 . 8 1 \%$ , $p { = } 0 . 0 2 $ ) (Fig. 6b). Our findings reveal a dual-regulatory architecture governing emotional dynamics in health crises. Mainstream media operates through an “authority amplification mechanism,” where institutional information produces heightened emotional responses via credibility amplification and attention concentration70. Conversely, We Media functions through an “attention diffusion mechanism,” wherein diverse, fragmented content moderates emotional reactivity by distributing cognitive processing across multiple narratives and engaging lateral evaluation processes71. 

More surprisingly, our media connection pattern analysis reveals a paradoxical buffering effect (Fig. 6c). For users connected to only one media type, risk reporting primarily promotes emotional intensification (mainstream media: $M  H$ increased by $2 0 . 9 0 \%$ , $p { < } 0 . 0 0 1$ ; We Media: $M  H$ increased by $4 . 3 0 \%$ , $\begin{array} { r }  p { = } 0 . 0 8 6 \ \end{array}$ ). However, users simultaneously connected to both media types exhibit a strikingly opposite response pattern: both media types’ risk reporting significantly promote transitions from high to medium arousal states (mainstream media: $H  M$ increased by $3 2 . 8 0 \%$ , $p { = } 0 . 0 0 6$ ; We Media: $H  M$ increased by $2 4 . 3 7 \%$ , $p { = } 0 . 0 3 4 $ ). This finding provides empirical evidence for the buffering mechanisms of social media use in collective resilience72 while extending the theoretical framework to a dual-channel media context of the “pluralistic media buffering effect”. Exposure to diverse information sources promotes cognitive reappraisal processes, enhancing users’ ability to integrate contradictory information. This finding aligns with Lynn et al.’s research on human information processing in complex networks73. It extends traditional information processing models by demonstrating how institutional hierarchy and network diversity create complementary regulation pathways: centralized channels establish emotional 

baselines while decentralized channels enable adaptive adjustments. This complementary relationship explains why media-diverse users demonstrate enhanced emotional stability—they benefit simultaneously from authoritative guidance and distributed processing flexibility, creating robust resilience to crisis information74. 

Node-level stability analysis provides compelling microscopic evidence for the dual-channel regulatory principle. Users connected to both media types exhibit significantly higher emotional stability indices $\mathrm { { S I } } = \ 0 . 6 4 4 5 )$ ) than those connected exclusively to mainstream media $\mathrm { \Delta S I } = 0 . 6 0 9 0$ , $\mathsf { p } = 0 . 0 0 1 5$ ) or We Media $( { \mathrm { S I } } = 0 . 6 0 7 3 ,$ p $= 0 . 0 0 0 8 )$ ), while no significant difference exists between single-channel users $( { \mathsf { p } } \ { = } $ 0.3507). This demonstrates that the buffering effect of diverse media exposure operates at the individual level rather than merely as an aggregate phenomenon. The enhanced stability of dual-connected nodes reveals a fundamental mechanism: when exposed to both institutional and social media channels, individuals develop more robust emotional regulation capacities through complementary information processing. Mainstream media provides authoritative framing that contextualizes risk, while We Media offers diverse perspectives that prevent emotional amplification. This complementarity creates a balanced information environment, stabilizing individual emotional states against perturbations. Importantly, this finding challenges the assumption that greater media exposure intensifies emotional responses to risk information. Instead, our results indicate that exposure diversity, rather than volume, determines emotional stability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/ea7784816ad98abbb23efff4cdd0e3015abebda40207539cf046a7e1fc5a22cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/bbea11e2ea6e8c4d3e91013cee9e00f3b14b1fa06f21cced05aaac4e75fda4df.jpg)



b


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/3960fac79bc6d1e0de1b6f059857a855d5c3295fd6398e04916726b9ad53d9bc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/cdc46225262f7eed8e0c0dcbcaa77a0f603a809c49e26269334ef07c84cb9d64.jpg)



Figure 6. Media exposure patterns and emotional stability across connection types. a, Differential impacts of mainstream media and We Media on emotional state transitions, showing stronger mainstream media effects on $\mathbf { M } \to \mathbf { H }$ transitions $. 1 7 . 5 5 \%$ , $\mathrm { p } { < } 0 . 0 1$ ) compared to We Media ( $4 . 8 4 \%$ , $\mathrm { p } { < } 0 . 0 5$ ). b, Heat map revealing distinct transition patterns across user groups: mainstream-


only users show significant $\mathbf { M } \to \mathbf { H }$ increases $( 2 0 . 9 0 \%$ , $\mathrm { p } { < } 0 . 0 5$ ), while dual-connected users exhibit enhanced $\mathrm { H } \to \mathbf { M }$ transitions under both media types (mainstream: $3 2 . 8 0 \%$ , We Media: $2 4 . 3 7 \%$ , $\mathrm { p } { < } 0 . 0 5 )$ ). c, Buffering effect analysis demonstrating how dual media exposure modulates emotional transitions, with increased $\mathrm { H } \to \mathbf { M }$ and decreased $\mathbf { M } \to \mathbf { H }$ probabilities compared to single-channel exposure. d, Node-level stability analysis showing significantly higher emotional stability in dualconnected users $\mathrm { ( S I = 0 . 6 4 4 5 { \pm } 0 . 0 0 8 2 }$ ) versus mainstream-only $\mathrm { \Delta S I = 0 . 6 0 9 0 { \pm } 0 . 0 0 7 5 }$ , $_ { \mathrm { p = 0 } . 0 0 1 5 }$ ) or We Media-only users $( \mathrm { S I } { = } 0 . 6 0 7 3 { \scriptstyle { \pm 0 . 0 0 6 9 } }$ , $\mathrm { p } { = } 0 . 0 0 0 8$ ). 

# Comparative Analysis of Media Channel Contributions

We conducted systematic model comparisons to evaluate the specific contributions of different information pathways to emotional dynamics in public health communication. Each model variant—removing mainstream media, removing We Media, or retaining only homophily effects—underwent identical parameter optimization procedures as the complete CSDAG model, ensuring fair comparison of predictive performance. By implementing these structured perturbations to the network architecture while maintaining consistent estimation methods, we can isolate the functional roles of specific media channels and peer effects in emotional state regulation. 

The CSDAG model demonstrates remarkable predictive accuracy across emotional states, achieving the highest consistency in tracking both high-arousal (0.615) and lowarousal (0.615) emotional transitions, with robust performance for middle-arousal states (0.538). This balanced accuracy across the emotional spectrum suggests that the model successfully captures the complex interplay between media influence and emotional dynamics. In striking contrast, the homophily-only model, which relies solely on peer interactions, shows negligible predictive power (trend consistency $\leq$ 0.077) across all emotional states, quantitatively demonstrating that peer influence alone cannot explain the observed emotional patterns. 

The comparative model analysis (Fig. 7) provides valuable insights into how different components of the CSDAG framework contribute to the observed emotional dynamics. By systematically varying model structures while maintaining consistent parameter estimates, we can evaluate the relative contribution of specific media channels to prediction accuracy. The variant without mainstream media connections (green lines) displays distinctive prediction patterns: consistently higher steady-state high-arousal emotions ${ \sim } 0 . 2 7$ vs. ${ \sim } 0 . 2 0$ in the complete model, Fig. 7a) and lower lowarousal states ${ \sim } 0 . 3 1$ vs. ${ \sim } 0 . 4 0$ , Fig. 7b). These systematic deviations suggest mainstream media’s substantial contribution to the regulation of emotional extremes, particularly in constraining high-arousal accumulation—a function that emerges from the interaction of multiple processes within the network structure. 

The no-We-Media variant (red lines) exhibits different characteristic deviations, most notably in its diminished ability to capture the rapid initial adaptation in middlearousal states (Fig. 8c, $\mathrm { t } { < } 1 0 $ ) and reduced responsiveness to temporal fluctuations. This pattern aligns with our framework’s identification of We Media’s role in facilitating rapid emotional adaptation through responsive content dynamics. 

The homophily-only model (orange lines) shows the largest deviations from empirical patterns, particularly in low-arousal states (Fig. 7b), demonstrating that social reinforcement mechanisms alone are insufficient to reproduce the observed emotional distributions without structured information inputs from media channels. 

Quantitatively, the trend consistency scores (Fig. 7d) confirm these observations: the complete CSDAG model achieves substantially higher prediction accuracy in higharousal (0.62 vs. 0.07-0.08) and low-arousal states (0.61 vs. 0.07-0.15), while all models perform similarly for middle-arousal states (0.53-0.54 vs. 0.07-0.15). This differential prediction pattern across emotional states highlights the complementary contributions of different network components to the overall system dynamics. 

These comparative results complement our state transition analysis by demonstrating how the interactions between network components generate the observed system-level behaviors, helping validate the theoretical framework of complementary regulatory functions performed by different media channels in emotional dynamics. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/4ae78b07fea3e2696e3316ec3de849232b0b38b71e6da300e52acf47417c7fba.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/5ac225accb1a76bf96c395b2b564d45dc38b3b8184a26be459408aaa1a5180c4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/99f238d32e7531a35b5977cde7e06a10ff73da2c84fd980cad1953ec49b1b56c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/87eca91ac9bbe3776ffd43af4d7ba88fb19e40b8f158da25d8ffeee78cd950a3.jpg)



CSDAG No WeMedia--Empirical



Figure 7. Comparative analysis of media influence on emotional dynamics through CSDAG framework variants. a-c, Time series comparison of emotional state predictions for high (a), middle (c), and low (b) arousal states across four models (CSDAG, homophily-only, no-mainstream, and no-We-Media) against empirical observations. d, Trend consistency scores across emotional states for all model variants. Solid lines represent model predictions while dashed lines with points show empirical data in time series plots (a-c). Colors distinguish between CSDAG (blue),


homophily-only (orange), no-mainstream (green), and no-We-Media (red) variants, maintaining consistent color coding across all panels. 

# Benchmarking CSDAG Against Classical Opinion Dynamics Models

To benchmark our CSDAG model, we employ two established opinion dynamics frameworks: the classical voter model and an enhanced variant. The classical voter model, with its binary state transitions and random neighbor imitation mechanism, provides a fundamental baseline for studying consensus formation in social networks75. The enhanced voter model extends this framework by incorporating differential influence strengths (2.0 for mainstream media, 1.5 for We Media, 1.0 for individuals) and state-dependent transition rules, better approximating the hierarchical nature of health information dissemination76. While both models capture basic contagion dynamics, they lack explicit mechanisms for emotional state transitions and media content adaptation. The classical voter model assumes uniform influence and random state copying, whereas the enhanced version introduces weighted influences and constrained state transitions but still operates without feedback mechanisms77. These limitations make them ideal baseline comparisons for evaluating the added value of our CSDAG framework’s dual-channel adaptation and emotional state coupling mechanisms. 

The comparative analysis between CSDAG and baseline models reveals distinct performance patterns across Long-COVID discourse dynamics. CSDAG demonstrates superior accuracy in emotional state predictions, particularly in capturing high-arousal emotional transitions (RMSE: 0.062 vs. 0.599 for Voter Model and 0.104 for Enhanced Voter). This advantage stems from CSDAG’s dual-channel adaptation mechanism, which explicitly accounts for the differential impacts of mainstream and We Media on emotional arousal levels, as evidenced in the time series plots (Fig. 8a-c). While capturing essential contagion dynamics, the classical voter model shows a significant deviation in tracking emotional states, particularly in high-arousal transitions (RMSE: 0.599), reflecting its limitations in modeling complex emotional cascades. 

For media risk reporting patterns, CSDAG achieves notably better predictions of We-media risk dynamics (RMSE: 0.137) compared to both baseline models (RMSE: 0.620 and 0.306, respectively). This improvement is particularly evident in capturing the characteristic fluctuations in We-media risk reporting (Fig. 8d), where CSDAG’s traffic-driven adaptation mechanism better reflects the platform’s responsive nature to audience engagement78. Incorporating hierarchical influence weights in the Enhanced Voter model fails to improve mainstream media risk predictions (RMSE: 0.455 vs. 0.294 for the classical model), highlighting the need for more sophisticated mechanisms to capture media-public sentiment dynamics. 

The overall performance comparison (Fig. 8f) quantitatively demonstrates CSDAG’s consistent advantage across all five measured dimensions. CSDAG significantly outperforms the voter model in emotional state predictions, reducing prediction errors by $8 9 . 6 \%$ (from 0.599 to 0.062) for high-arousal states and $8 1 . 6 \%$ 

(from 0.377 to 0.069) for middle-arousal states. These improvements validate our theoretical framework’s emphasis on bidirectional adaptation between media content and public sentiment, particularly in capturing the distinct roles of mainstream and We Media in emotional contagion processes. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/cdb60e2ca2a9f3a966de81d74b338c20b55147d73691d9b382523454c04945cd.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/7492d77cd92df7efc1d12fcefc051b9cb5f2d94a62cae805ef70b28f73ceb345.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/b8789a7498b38bf2e7ca259c6c2bec6f11dc248a59a61941fa6bc693b8226e12.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/4df2706fa8850793b28c64ed4c3c97d6026ad2cc06a38cd555d83f3ed9cf3850.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/86ac8b176fc81d59c42bf9b16375d51f62f899318dde1346ab56eaea8c591712.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/b3d881a89aad82bebc661a8e1d251a34ac1900201c26d71d5c29d60b8b5d4a6c.jpg)



Figure 8. Comparative analysis of model performance in predicting emotional states and media risk dynamics. a-c, Time series comparison of emotional state predictions across three


models (CSDAG, Voter Model, Enhanced Voter) against empirical data for high (a), middle (b), and low (c) arousal states. d-e, Model predictions for risk reporting patterns in We Media (d) and mainstream media (e) compared with empirical observations. f, Root Mean Square Error (RMSE) comparison across five key metrics, demonstrating relative prediction accuracy of the three models. Solid lines represent model predictions while dashed lines show empirical data in time series plots (a-e). Bar colors distinguish between CSDAG (pink), Voter Model (blue), and Enhanced Voter Model (green) predictions. 

# Discussion

Understanding how different types of media content shape public emotional responses during health crises has become increasingly critical. While extensive research has examined media’s influence on public sentiment, the mechanisms through which mainstream and social media content modulate emotional dynamics remain poorly understood. Traditional approaches have largely focused on simple exposure effects, overlooking the complex interplay between media content and emotional state transitions. This limitation is particularly evident in Long-COVID discourse, where persistent health uncertainties create unique patterns of emotional response to media content. Our study addresses this gap by developing a comprehensive framework that captures the bidirectional relationships between media channels and public sentiment, leveraging network analysis to reveal how different information pathways shape collective emotional responses to health crises. 

Our study makes three key contributions to understanding how media content shapes public emotional dynamics during health crises. First, we demonstrate that emotional responses emerge from a dual-pathway regulatory system rather than simple exposure effects. Mainstream media operates through an “authority amplification mechanism,” where official risk reporting significantly increases transitions toward higher arousal states due to concentrated attention and heightened credibility. Conversely, We Media functions through an “attention diffusion mechanism,” where diverse, fragmented content creates multidirectional effects that distribute attention and often counterbalance extreme reactions. Second, our state transition analysis reveals that media connection patterns fundamentally alter emotional regulation. Most significantly, users exposed to both media types demonstrate enhanced emotional stability rather than amplified reactions, with significant increases in transitions from high to middle arousal states. This counterintuitive finding challenges the assumption that greater exposure to risk information inevitably intensifies negative emotions, suggesting instead that media diversity enables more sophisticated emotional processing through complementary information pathways. Third, our CSDAG framework demonstrates that accurate prediction of public emotional responses requires modeling the dynamic interplay between media content and community-level feedback. The model's robust performance $\mathrm { R M S E } { = } 0 . 1 2 5$ ) emerges from capturing 

how mainstream media provides structural stability through consistent content patterns while We Media enables adaptive responsiveness through feedback-sensitive content dynamics. Counterfactual analysis reveals systematic degradation in predictive power when either channel is removed, confirming that emotional dynamics emerge from their synergistic interaction rather than independent influences. These findings indicate that effective health crisis communication requires shifting from channel-specific strategies to integrated approaches that balance authoritative guidance with adaptive feedback mechanisms. The complementary functions of different media types create a selfregulating system capable of both activating appropriate concern and preventing emotional extremes—a crucial insight for managing communication during health emergencies where both information deficits and emotional fatigue present significant challenges. 

This research acknowledges several important limitations. First, our model focuses exclusively on the dynamics between individuals and media channels, overlooking other influential factors such as community bonds and interactions with internet strangers, which can significantly impact emotional states79. While this simplification enables us to isolate and analyze media-emotion coupling mechanisms, it may underestimate the complexity of social influence in emotional contagion. Second, the potential for sampling bias exists due to the sensitive nature of Long-COVID discourse, where specific comments may be censored or hidden, affecting the external validity of our model. Although our large dataset helps mitigate this concern, the systematic exclusion of certain viewpoints remains a limitation. Third, our static network approach may not fully capture the temporal evolution of information pathways, particularly during emergent events or social campaigns80. This limitation is especially relevant given the dynamic nature of social media platforms and their rapid adaptation to changing circumstances. Fourth, our equal-weight assumption for media outlets in the CSDAG framework oversimplifies the real-world dynamics of information propagation. Future research should incorporate engagement metrics such as click rates, likes, and shares to assign differentiated influence weights to various media outlets, providing a more nuanced representation of information dissemination patterns. 

# Materials and Methods

This section presents our methodological framework for analyzing the complex dynamics of Long-COVID discourse. The CSDAG model integrates network structure, agent dynamics, and empirically calibrated parameters to capture the interplay between media content and public sentiment. Through automated content analysis of Weibo data, we extract latent risk signals and emotional responses, enabling the simulation of interactions among mainstream media, self-media, and the public. Our agent-based modeling approach, refined through Bayesian parameter estimation, reveals the causal 

mechanisms underlying the co-evolution of risk communication and emotional dynamics in public health discourse. 

# Automated sentiment analysis

The emotional dynamics in health crisis communication emerge from complex psychological processes that shape public responses to threat information. Our classification framework builds upon psychological arousal theory81 and its recent empirical validations in digital communication contexts. Recent studies have revealed how emotional states fundamentally influence information processing and behavioral responses during crises, with arousal levels serving as a key predictor of social transmission patterns82. 

Drawing from this theoretical foundation, we developed a systematic approach to emotion classification that combines lexicon-guided sample selection with advanced machine learning techniques. The reliability of classification critically depends on the quality of training samples, which we ensure through a comprehensive confidence scoring mechanism. This scoring system evaluates emotional expression quality through weighted assessment of direct emotional terms (weight $= 1 . 0 $ ), emotional intensifiers (weight $= 0 . 5$ ), and contextual markers (weight $= 0 . 3$ ). The confidence score for each text segment is computed as: 

$$
C _ {i} = \frac {N _ {\text {p r i m a r y}}}{N _ {\text {t o t a l}}} \cdot D _ {i} \cdot L _ {i}
$$

where $N _ { p r i m a r y }$ represents primary emotion category terms, $N _ { { _ { t o t a l } } }$ indicates total emotional terms, $D _ { i }$ captures category distinctiveness, and $L _ { i }$ reflects length optimization (see supplementary section 2). 

Building upon this foundation, we implemented a rigorous annotation protocol with three independent annotators, each having background in communication studies. The annotation process maintained strict independence through a blind review system, with final classifications determined through majority voting. Inter-rater reliability analysis revealed strong agreement $( \kappa = 0 . 8 2 )$ , with particularly high consistency in identifying high-arousal states $\left( \kappa = 0 . 8 7 \right)$ ). 

The computational implementation builds upon the DistilBERT architecture, optimized specifically for emotion recognition in health crisis discourse. The model integrates a specialized preprocessing pipeline for social media expressions while maintaining the theoretical grounding of our three-level classification system. Training employed our complete dataset of 58,269 samples, naturally distributed across communication channels: mainstream media $( 1 2 . 8 \% )$ , social media $( 3 6 . 8 \% )$ , and public posts $( 5 0 . 4 \% )$ . Model validation through ten-fold cross-validation demonstrated consistent classification accuracy of $8 2 . 1 \%$ , indicating robust performance across 

different data partitions (See supplementary section 2). 

Beyond emotion classification, our framework also addresses the critical task of risk signal assessment in media content. The classification of risk reporting patterns in mainstream and social media channels represents a distinct yet complementary analytical dimension to our emotion analysis. This dual-classification approach enables comprehensive understanding of how different media channels modulate public emotional responses through their risk reporting strategies. 

For mainstream media content, our classification framework distinguishes between risk-focused and non-risk coverage patterns. The model achieves robust accuracy $( 9 0 . 1 \% )$ in identifying risk-related content, with particular sensitivity to formal risk assessment language and expert citations. This high performance reflects the relatively standardized nature of risk reporting in institutional media channels.Social media (Wemedia) risk classification presents unique challenges due to the diverse and informal nature of risk discourse in these channels. Our adapted classification approach accounts for platform-specific expression patterns and achieves strong performance $( 8 6 . 0 \% )$ in distinguishing risk-focused content. This classification captures both explicit risk statements and implicit risk signals common in social media discourse (See supplementary section 3). 

The integration of these classification tasks - emotion states and risk signals - provides a comprehensive analytical framework for understanding the dynamic interplay between media content and public emotional responses during health crises. 

# Network Structure and State Space

Within this framework, media nodes (M and W ) occupy binary states Lm , Lw ∈ {norisk , risk }, representing their stance on Long-COVID risks. Individual $L _ { t } ^ { m } , L _ { t } ^ { w } \in \{ \mathrm { n o r i s k } , \mathrm { r i s k } \} ^ { \mathrm { } }$ users, however, inhabit a more nuanced state space Ls=( E , I )∈ {H , M , L}× [0 ,1], $\begin{array} { r } { L _ { t } ^ { s } \mathbf { = } \{ E _ { t } , I _ { t } \} \in \{ H , M , L \} \times \{ 0 , 1 \} } \end{array}$ 

where $E _ { t }$ captures discrete arousal levels (High/Middle/Low) while $I _ { \ d _ { t } }$ quantifies continuous emotional intensity. This hybrid state space, reminiscent of complex systems with discrete energy levels and continuous wavefunctions, enables smooth transitions between emotional states through intensity variations. The continuous weights $S _ { i j }$ in user-user interactions mirror interaction strengths in spin systems, allowing us to model the subtle dynamics of emotional contagion through social ties. 

The networks layered architecture naturally accommodates the observed asymmetric influence patterns: mainstream media, positioned at the top layer, exerts authoritative influence through direct connections to users, while We Media occupies an intermediate position, balancing official narratives with audience engagement. This structure proves crucial in capturing the unique characteristics of China’s media ecosystem during public health crises. For visualization clarity, Fig. 2 presents a $1 0 \%$ random sample of the complete network. Detailed network statistics and initialization protocols are provided in Supplementary Section 1. 

# Content Module

The content module in CSDAG captures the dynamic interplay between media risk signals and public sentiment through a statistical physics framework analogous to spin systems83. The module operates through two coupled mechanisms: mainstream media’s social stability-oriented regulation and We Media’s traffic-driven adaptation. 

# 1. Mainstream content dynamics

For mainstream media nodes, we introduce an adaptive regulatory mechanism that combines local sentiment-driven transitions with a global regulatory field. The state transition probability is modeled as: 

$$
P _ {m} \left(L _ {t} ^ {i} \rightarrow L _ {t + 1} ^ {i}\right) = \frac {1 - \exp (- \alpha d) + \sigma (g _ {m})}{2} \tag {2}
$$

where $d$ quantifies the local sentiment disparity: 

$$
d = \left\{ \begin{array}{l l} 2 n _ {h} - \left(n _ {m} + n _ {l}\right) & \text {i f} n _ {h} > n _ {m}, n _ {l} \\ 2 n _ {l} - \left(n _ {m} + n _ {h}\right) & \text {i f} n _ {l} > n _ {m}, n _ {h} \\ 2 n _ {m} - \left(n _ {l} + n _ {h}\right) & \text {i f} n _ {m} > n _ {h}, n _ {l} \end{array} \right. \tag {3}
$$

Here, $n _ { h } , n _ { m } ,$ , and $n _ { l }$ represent the populations of high, middle, and low arousal states in the node’s neighborhood respectively. The regulatory field strength $g _ { m }$ is transformed through a sigmoid function $\sigma ( g _ { m } )$ that maps the field to (0,1), enabling continuous modulation of the transition probability. 

This formulation reveals that mainstream media’s regulatory dynamics operate through a dual-scale mechanism analogous to physical systems near criticality84. At the microscopic level, sentiment disparities between connected nodes drive state transitions through Boltzmann-like probabilities, naturally pushing the system toward local equilibrium. These transitions are simultaneously modulated by a global regulatory field that adapts its strength based on the system’s information entropy73, enabling responsive control of large-scale instabilities. The framework provides a natural explanation for how mainstream media maintains social stability - local sentiment fluctuations are contained through neighbor interactions, while systemic risks are managed through entropy-mediated feedback that strengthens regulatory influence when disorder increases. 

# 2. We Media content dynamics

We Media nodes follow a dual-driven transition mechanism: 

$$
\begin{array}{c}P _ {w} \left(L _ {t} ^ {w _ {i}} \rightarrow L _ {t + 1} ^ {w _ {i}}\right) = P _ {v 1} P _ {v 2}\\(4)\end{array}
$$

where $P _ { \nu 1 } { = } 1 { - } \exp \left( { - } \beta H \right)$ captures the response to public sentiment entropy $H$ , and the alignment probability with the overall media landscape is given by: 

$$
P _ {v 2} = \frac {\tanh  \left(N _ {N R} - N _ {R}\right) + 1}{2} \tag {5}
$$

Here, $N _ { { } _ { N R } }$ and $N _ { \scriptscriptstyle R }$ represent the total number of non-risk and risk signals across all media channels respectively. This formulation reveals how We Media balances information entropy and systemic risk signals in content production. The entropy-based component $P _ { v 1 }$ quantifies We Media’s response to public sentiment diversity, where higher entropy indicates greater uncertainty in public emotional states. Meanwhile, $P _ { v 2 }$ captures We Media’s tendency to align with the dominant risk narrative in the broader media environment. This dual mechanism enables We Media to adaptively moderate between audience engagement and systemic stability - when public sentiment becomes highly uncertain (large $H$ ), We Media increases content production to engage audiences, while simultaneously maintaining alignment with the prevailing media consensus through the nonlinear tanh function. The interplay between these competing drives creates a self-organizing regulatory mechanism that helps maintain the stability of the overall information ecosystem. 

# Sentiment Module

Individual sentiment dynamics in the CSDAG network are governed by dual fields - media information and local sentiment distribution. The transition probability for an individual node is described by the intensity function: 

$$
I _ {i, t} = I _ {i, t - 1} - \frac {\tanh  (\theta \cdot d _ {1}) + \tanh  (\sigma \cdot d _ {2}) + 2}{4} \tag {6}
$$

where $I _ { i , t }$ represents the sentiment intensity of node i at time t. The field differences $d _ { 1 }$ and $d _ { 2 }$ capture media influence and social pressure respectively. For media influence, $d _ { 1 }$ measures the imbalance between risk and non-risk signals $( N _ { R } - N _ { N R } )$ , with its sign determined by the current emotional state: negative for high-arousal states, positive for 

low-arousal states, and absolute difference for middle-arousal states. The social pressure field $d _ { 2 }$ quantifies the tension between an individual’s current emotional state and the surrounding sentiment distribution through: 

$$
d _ {2} = n _ {\text {c o m p e t e n g}} - n _ {\text {c u r r e n t}} \tag {7}
$$

where ncompeting represents the sum of competing emotional states (middle and low arousal for high-arousal individuals, high and low for middle-arousal, and high and middle for low-arousal), while $n _ { c u r r e n t }$ reflects the population in the individual’s present emotional state. 

For middle-arousal states, which serve as a transition zone, the directional probability $P _ { s }$ determines the likelihood of transitioning toward higher arousal states: 

$$
P _ {s} = \frac {1 + \tanh  \left(N _ {R} - N _ {N R}\right)}{2} \tag {8}
$$

The transition occurs when intensity falls below a random threshold, with parameter $\zeta$ balancing the influence between media guidance and social conformity. This formulation reveals how individual emotional states evolve through the interplay of emotional inertia $( I _ { i , t - 1 } )$ , nonlinear response to information fields, and statedependent social influence - a mechanism that naturally captures the observed patterns of emotional contagion in social networks. 

# Calibration

We employed Approximate Bayesian Computation (ABC) with Sequential Monte Carlo sampling to calibrate the CSDAG model parameters. The ABC framework iteratively refines parameter estimates by minimizing the root mean square error (RMSE) between simulated and empirical trajectories across five key metrics: emotional states $( n _ { { \scriptscriptstyle H } } , n _ { { \scriptscriptstyle M } } , n _ { { \scriptscriptstyle L } } )$ and media risk signals $( R _ { m } , R _ { w } )$ . For each iteration, we evaluated ${ 1 0 } ^ { 4 }$ parameter combinations drawn from prior distributions using Latin Hypercube sampling, accepting the top $2 5 \%$ performers for subsequent refinement. The parameter space $\Theta$ encompasses both continuous parameters and discrete choices: 

$$
\Theta = \left\{ \begin{array}{l} \theta_ {c} \in R ^ {d} \end{array} \right\} \times \left\{\theta_ {d} \in D \right\} \tag {9}
$$

where $\theta _ { c }$ represents continuous parameters bounded by physical constraints, and $\theta _ { d }$ represents discrete choices from the set $D$ of possible functional forms. 

The ABC algorithm iteratively refines parameter estimates by minimizing the distance between simulated and empirical resilience trajectories85. For each iteration t, we generate $N$ particles (parameter sets) $\{ \theta _ { i } ^ { t } \} _ { i = 1 } ^ { N }$ and evaluate their fitness through a distance metric: 

$$
\rho (\theta) = \sqrt {\frac {1}{T} \sum_ {t = 1} ^ {T} \square (s _ {t} (\theta) - s _ {t} ^ {i}) ^ {2}}
$$

where $s _ { t } ( \theta )$ represents the simulated system state at time $t$ using parameters $\theta$ , and $\textit { i }$ represents the empirical observations. $s _ { t }$ 

The algorithm adaptively updates parameter ranges based on accepted particles that satisfy $\rho \left( \theta \right) < \epsilon _ { t } ,$ , where $\epsilon _ { t }$ is a dynamically adjusted threshold: 

$$
\epsilon_ {t} = Q _ {0. 2 5} \left(\rho \left(\theta_ {i} ^ {t}\right) _ {i = 1} ^ {N}\right)
$$

For continuous parameters, the ranges are updated using an interquartile-based approach: 

$$
\left[ l _ {t + 1}, u _ {t + 1} \right] = \left[ Q _ {1} - \alpha \cdot I Q R, Q _ {3} + \alpha \cdot I Q R \right]
$$

where $Q _ { 1 } , Q _ { 3 }$ are the first and third quartiles of accepted parameters, IQR is the interquartile range, and $\alpha$ is an expansion factor (set to 1.5 in our implementation). 

The algorithm incorporates parameter sensitivity analysis through local perturbation: 

$$
S _ {j} = \frac {1}{2 \Delta} \sum_ {k = 1} ^ {2} \square \left| \rho (\theta + (- 1) ^ {k} \Delta e _ {j}) - \rho (\theta) \right|
$$

where $S _ { j }$ quantifies the sensitivity of parameter $j , \Delta$ represents the perturbation magnitude, and $e _ { j }$ is the unit vector in the $j \cdot$ -th dimension. 

This calibration framework demonstrates robust convergence across different disaster scenarios. The framework’s effectiveness is validated through out-of-sample testing on held-out disaster events, showing consistent performance in capturing both immediate impact dynamics and long-term recovery trajectories. 

The optimal parameter set $\theta ^ { \dot { \zeta } }$ achieves an RMSE of 0.125, with values: mainstream 

media response $\alpha { = } 3 9 . 6 2$ , We Media adaptability $\beta { = } 0 . 8 6$ , emotional transition rate $\theta { = } 0 . 0 1$ , social influence strength $\sigma { = } 3 3 . 8 5$ , memory decay $\zeta = 0 . 5 7$ , and regulatory field strength $\gamma { = } 9 6 . 8 8$ . These calibrated values reveal distinct mechanistic features: the high $\alpha$ and $\sigma$ values indicate strong institutional media influence and social conformity effects, while the moderate $\beta$ suggests systematic audience adaptation. The high γ coupled with low $\theta$ points to a regulatory system that maintains stability through continuous, small-scale emotional adjustments rather than abrupt transitions. 

The posterior distributions demonstrate parameter convergence within 6-8 iterations, with particularly stable estimates for $\beta$ and ζ , reflecting their crucial roles in system stability (Fig. 5). This calibration performance validates our theoretical framework’s capacity to capture the coupled dynamics between media content adaptation and public emotional response, particularly during critical transition periods. 

# State Transition Analysis

To decipher how different media channels modulate public emotional responses during health crises, we developed a multi-level analytical framework that quantifies both state-level transitions and node-level stability patterns. This approach enables precise measurement of emotional dynamics across varied media exposure patterns. 

# 1. State Transition Probability Analysis

We first established a probabilistic framework to measure emotional state transitions under varying risk reporting intensities. For each media channel $c \in \{ m , w \}$ (mainstream or We Media), we defined high-risk periods using the 75th percentile threshold $\theta _ { c }$ of the risk reporting time series $R _ { c }$ : 

$$
H _ {c} = \left\{t \mid R _ {c} (t) \geq \theta_ {c} \right\}, L _ {c} = \left\{t \mid R _ {c} (t) <   \theta_ {c} \right\} \tag {14}
$$

where $H _ { c }$ and $L _ { c }$ represent high-risk and low-risk periods, respectively. For each emotional state pair $\left( i , j \right) \in \left\{ L , M , H \right\} ^ { 2 }$ , we calculated the transition probability under risk condition $r \in \{ h i g h , l o w \}$ : 

$$
P _ {r} ^ {c} (i \rightarrow j) = \frac {N _ {r} ^ {c} (i \rightarrow j)}{\sum_ {k \in \{L , M , H \}} \square N _ {r} ^ {c} (i \rightarrow k)} \tag {15}
$$

where N c ( i → j ) $N _ { r } ^ { c } ( i  j )$ counts observed transitions from state  to  during risk condition 

r for media channel $c$ . This formulation captures how risk information modulates emotional state transitions. 

To establish statistical significance, we implemented bootstrap testing (200 resamples) to compute confidence intervals for the difference in transition probabilities: 

$$
\Delta_ {i j} ^ {c} = P _ {h i g h} ^ {c} (i \rightarrow j) - P _ {l o w} ^ {c} (i \rightarrow j)
$$

We identified key transitions most sensitive to risk reporting: moderate-to-high arousal $( M \to H )$ , low-to-moderate arousal $( L \to M )$ , and high-to-moderate arousal ( $H  M$ ), following psychological arousal theory and our empirical observation. 

# 2. Connection Pattern Analysis and Buffering Effect Quantification

We categorized user nodes based on their media connection patterns: 

 Mainstream-only: Users connected exclusively to mainstream media nodes 

 We Media-only: Users connected exclusively to We Media nodes 

 Dual-connected: Users with connections to both media types 

For each group g ∈{mainstream , wemedia , dual }, we calculated separate transition matrices under high and low risk conditions, enabling systematic comparison of media effects across different connection topologies. This differentiated analysis revealed how network structure modulates emotional responses to risk information. 

# 3. Node Stability Index

To quantify long-term emotional resilience, we developed a Node Stability Index (SI). For each user node v, we calculated: 

$$
S I _ {v} = \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \square 1 \left(s _ {t} ^ {v} = s _ {t + 1} ^ {v}\right)
$$

where $T$ represents the total time steps, $s _ { t } ^ { v }$ st denotes the emotional state of node  at time $t$ , and $1 ( \cdot )$ is the indicator function. This index measures the proportion of time steps during which a node maintains its emotional state, providing a robust metric of emotional stability. 

Statistical comparisons between connection pattern groups employed two-tailed ttests with appropriate significance thresholds $\scriptstyle ( \alpha = 0 . 0 5 )$ . This rigorous analytical 

framework enabled us to identify the differential buffering effects of varied media exposure patterns on emotional dynamics, revealing the underlying mechanisms of dual-channel emotional regulation in crisis communication. 

The methodology steps are detailed in Fig. 9. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/ed4d9c28-7492-48d4-991b-529dd68e05e7/c7f3fc0c01386923b2af5f52f9bad4e203521eb1ad3abefb2f9541c39b951f09.jpg)



Figure 9. Overview of Research Methodology. This figure depicts our three-stage approach, starting with data initialization, proceeding to the simulation phase, and culminating in model evaluation, where we validate our hypotheses.


# Code availability

The code used for data analysis and model simulation in this study is available in the same GitHub repository: https://github.com/xxx/CSDAG. This includes implementation of the Content-Sentiment Directed Approval Graph (CSDAG) model and associated analysis scripts. 

# Data availability

The network data, empirical data, and calibrated parameter data that support the findings of this study are publicly available in the GitHub repository: https://github.com/xxx/CSDAG. The raw social media data from Sina Weibo used in this study are not publicly available due to privacy protection requirements but are 

available from the corresponding author upon reasonable request with appropriate data protection agreements. 

# References



1. Pangallo, M. et al. The unequal effects of the health–economy trade-off during the COVID-19 pandemic. Nat. Hum. Behav. 8, 264–275 (2023). 





2. Liu, Z., Wu, J., Wu, C. Y. H. & Xia, X. Shifting sentiments: analyzing public reaction to COVID-19 containment policies in Wuhan and Shanghai through Weibo data. Humanit. Soc. Sci. Commun. 11, 1–13 (2024). 





3. Kim, S. et al. Short- and long-term neuropsychiatric outcomes in long COVID in South Korea and Japan. Nat. Hum. Behav. 8, 1530–1544 (2024). 





4. Rubin, R. As Their Numbers Grow, COVID-19 “Long Haulers” Stump Experts. JAMA 324, 1381–1383 (2020). 





5. Baig, A. M. Chronic COVID syndrome: Need for an appropriate medical terminology for long-COVID and COVID long-haulers. J. Med. Virol. 93, 2555–2556 (2021). 





6. Sudre, C. H. et al. Attributes and predictors of long COVID. Nat. Med. 27, 626–631 (2021). 





7. Davis, H. E., McCorkell, L., Vogel, J. M. & Topol, E. J. Long COVID: major findings, mechanisms and recommendations. Nat. Rev. Microbiol. 21, 133–146 (2023). 





8. Crook, H., Raza, S., Nowell, J., Young, M. & Edison, P. Long covid—mechanisms, risk factors, and management. BMJ 374, (2021). 





9. Vartanian, K. et al. Integrating patient-reported physical, mental, and social impacts to classify long COVID experiences. Sci. Rep. 13, 16288 (2023). 





10. Byrne, E. A. Understanding Long Covid: Nosology, social attitudes and stigma. Brain. Behav. Immun. 99, 17–24 (2022). 





11. Brüssow, H. & Timmis, K. COVID-19: long covid and its societal consequences. Environ. Microbiol. 23, 4077–4091 (2021). 





12. Chen, X. & Yik, M. The Emotional Anatomy of the Wuhan Lockdown: Sentiment Analysis Using Weibo Data. JMIR Form. Res. 6, e37698 (2022). 





13. Sher, L. Post-COVID syndrome and suicide risk. QJM Mon. J. Assoc. Physicians 114, 95–98 (2021). 





14. Yong, E. Opinion | Reporting on Long Covid Taught Me to Be a Better Journalist. The New York Times (2023). 





15. Smith, P. et al. Post COVID-19 condition and its physical, mental and social implications: protocol of a 2-year longitudinal cohort study in the Belgian adult population. Arch. Public Health Arch. Belg. Sante Publique 80, 151 (2022). 





16. Cunningham, S. Popular media as public ‘sphericules’ for diasporic communities. Int. J. Cult. Stud. 4, 131–147 (2001). 





17. Fernandez, G. et al. Social Network Analysis of COVID-19 Sentiments: 10 Metropolitan Cities in Italy. Int. J. Environ. Res. Public. Health 19, 7720 (2022). 





18. Crocamo, C. et al. Surveilling COVID-19 Emotional Contagion on Twitter by Sentiment Analysis. Eur. Psychiatry 64, e17 (2021). 





19. Iglesias-Sánchez, P. P., Vaccaro Witt, G. F., Cabrera, F. E. & Jambrino-Maldonado, C. The Contagion of Sentiments during the COVID-19 Pandemic Crisis: The Case of Isolation in Spain. Int. J. Environ. Res. Public. Health 17, 5918 (2020). 





20. Li, S., Wang, Y., Xue, J., Zhao, N. & Zhu, T. The Impact of COVID-19 Epidemic Declaration on 





Psychological Consequences: A Study on Active Weibo Users. Int. J. Environ. Res. Public. Health 17, 2032 (2020). 





21. Granovetter, M. S. The Strength of Weak Ties. Am. J. Sociol. 78, 1360–1380 (1973). 





22. McPherson, M., Smith-Lovin, L. & Cook, J. M. Birds of a Feather: Homophily in Social Networks. Annu. Rev. Sociol. 27, 415–444 (2001). 





23. Kossinets, G. & Watts, D. J. Origins of Homophily in an Evolving Social Network. Am. J. Sociol. 115, 405–450 (2009). 





24. Khanam, K. Z., Srivastava, G. & Mago, V. The homophily principle in social network analysis: A survey. Multimed. Tools Appl. 82, 8811–8854 (2023). 





25. Simon, M., Welbers, K., C. Kroon, A. & Trilling, D. Linked in the dark: A network approach to understanding information flows within the Dutch Telegramsphere. Inf. Commun. Soc. 26, 3054–3078 (2023). 





26. Fan, R., Xu, K. & Zhao, J. An agent-based model for emotion contagion and competition in online social media. Phys. Stat. Mech. Its Appl. 495, 245–259 (2018). 





27. Li, S., Liu, Z. & Li, Y. Temporal and spatial evolution of online public sentiment on emergencies. Inf. Process. Manag. 57, 102177 (2020). 





28. Hao, X., An, H., Zhang, L., Li, H. & Wei, G. Sentiment Diffusion of Public Opinions about Hot Events: Based on Complex Network. PLOS ONE 10, e0140027 (2015). 





29. Zhang, L., Li, H. & Chen, K. Effective Risk Communication for Public Health Emergency: Reflection on the COVID-19 (2019-nCoV) Outbreak in Wuhan, China. Healthcare 8, 64 (2020). 





30. Wu, Y., Xiao, H. & Yang, F. Government information disclosure and citizen coproduction during COVID-19 in China. Governance 35, 1005–1027 (2022). 





31. McCOMBS, M. E. & SHAW, D. L. THE AGENDA-SETTING FUNCTION OF MASS MEDIA*. Public Opin. Q. 36, 176–187 (1972). 





32. Lasswell, H. The structure and function of communication in society. in (2007). 





33. Entman, R. M. Framing: Toward Clarification of a Fractured Paradigm. J. Commun. 43, 51–58 (1993). 





34. Scheufele, D. Framing as a theory of media effects. J. Commun. 49, 103–122 (1999). 





35. Stockmann, D. Media Commercialization and Authoritarian Rule in China. (Cambridge University Press, Cambridge, 2012). doi:10.1017/CBO9781139087742. 





36. Barbieri, N., Bonchi, F. & Manco, G. Topic-Aware Social Influence Propagation Models. in 2012 IEEE 12th International Conference on Data Mining 81–90 (2012). doi:10.1109/ICDM.2012.122. 





37. Li, Y., Hills, T. & Hertwig, R. A brief history of risk. Cognition 203, 104344 (2020). 





38. Wu, G., Deng, X. & Liu, B. Using fear appeal theories to understand the effects of location information of patients on citizens during the COVID-19 pandemic. Curr. Psychol. 42, 17316– 17328 (2023). 





39. Nabi, R. L. Emotional Flow in Persuasive Health Messages. Health Commun. 30, 114–124 (2015). 





40. Nabi, R. L. & Green, M. C. The Role of a Narrative’s Emotional Flow in Promoting Persuasive Outcomes. Media Psychol. 18, 137–162 (2015). 





41. Carretié, L. Exogenous (automatic) attention to emotional stimuli: A review. Cogn. Affect. Behav. Neurosci. 14, 1228–1258 (2014). 





42. Wang, P., Shi, H., Wu, X. & Jiao, L. Sentiment Analysis of Rumor Spread Amid COVID-19: Based on Weibo Text. Healthcare 9, 1275 (2021). 





43. Bavel, J. J. V. et al. Using social and behavioural science to support COVID-19 pandemic response. Nat. Hum. Behav. 4, 460–471 (2020). 





44. Molyneux, L. What journalists retweet: Opinion, humor, and brand development on Twitter. Journalism 16, 920–935 (2015). 





45. Kozyreva, A., Lorenz-Spreen, P., Hertwig, R., Lewandowsky, S. & Herzog, S. M. Public attitudes towards algorithmic personalization and use of personal data online: evidence from Germany, Great Britain, and the United States. Humanit. Soc. Sci. Commun. 8, 117 (2021). 





46. van Dijck, J. & Poell, T. Understanding Social Media Logic. SSRN Scholarly Paper at https://papers.ssrn.com/abstract=2309065 (2013). 





47. Jiang, Y. ‘Reversed agenda-setting effects’ in China Case studies of Weibo trending topics and the effects on state-owned media in China. J. Int. Commun. 20, 168–183 (2014). 





48. McGinty, E. E., Presskreischer, R., Han, H. & Barry, C. L. Psychological Distress and Loneliness Reported by US Adults in 2018 and April 2020. JAMA 324, 93 (2020). 





49. Myrick, J. G. & Nabi, R. L. Fear Arousal and Health and Risk Messaging. in Oxford Research Encyclopedia of Communication (2017). doi:10.1093/acrefore/9780190228613.013.266. 





50. Stijačić, M. P., Mišić, K. & Đurđević, D. F. Flattening the curve: COVID-19 induced a decrease in arousal for positive and an increase in arousal for negative words. Appl. Psycholinguist. 44, 1069–1089 (2023). 





51. Lekkas, D., Gyorda, J. A., Price, G. D., Wortzman, Z. & Jacobson, N. C. Using the COVID-19 Pandemic to Assess the Influence of News Affect on Online Mental Health-Related Search Behavior Across the United States: Integrated Sentiment Analysis and the Circumplex Model of Affect. J. Med. Internet Res. 24, e32731 (2022). 





52. Dolinšek, Š. et al. The role of mental well-being in the effects of persuasive health messages: A scoping review. Soc. Sci. Med. 353, 117060 (2024). 





53. Lang, A., Park, B., Sanders-Jackson, A. N., Wilson, B. D. & Wang, Z. Cognition and Emotion in TV Message Processing: How Valence, Arousing Content, Structural Complexity, and Information Density Affect the Availability of Cognitive Resources. Media Psychol. 10, 317– 338 (2007). 





54. Gudykunst, W. D. Anxiety/uncertainty management (AUM) theory: Current status. in Intercultural communication theory 8–58 (Sage Publications, Inc, Thousand Oaks, CA, US, 1995). 





55. Martinelli, N. et al. Time and Emotion During Lockdown and the Covid-19 Epidemic: Determinants of Our Experience of Time? Front. Psychol. 11, (2021). 





56. Heffner, J., Vives, M.-L. & FeldmanHall, O. Emotional responses to prosocial messages increase willingness to self-isolate during the COVID-19 pandemic. Personal. Individ. Differ. 170, 110420 (2021). 





57. Changing emotions in the COVID-19 pandemic: A four-wave longitudinal study in the United States and China. Soc. Sci. Med. 285, 114222 (2021). 





58. Zhao, L. et al. Sentiment contagion in complex networks. Phys. Stat. Mech. Its Appl. 394, 17–23 (2014). 





59. Kozitsin, I. V. A general framework to link theory and empirics in opinion formation models. Sci. Rep. 12, 5543 (2022). 





60. Lewis, K., Gonzalez, M. & Kaufman, J. Social selection and peer influence in an online social network. PNAS Proc. Natl. Acad. Sci. U. S. Am. 109, 68–72 (2012). 





61. Cosme, D. et al. Message self and social relevance increases intentions to share content: Correlational and causal evidence from six studies. J. Exp. Psychol. Gen. 152, 253–267 (2023). 





62. Boston University, Watts, S., Zhang, W., & University of Massachusetts Boston, USA. Capitalizing on Content: Information Adoption in Two Online communities. J. Assoc. Inf. Syst. 9, 73–94 (2008). 





63. Couldry, N. & Turow, J. Advertising, Big Data and the Clearance of the Public Realm: Marketers’ New Approaches to the Content Subsidy. Int. J. Commun. 8, 17 (2014). 





64. Casini, L. & Manzo, G. Agent-Based Models and Causality : A Methodological Appraisal. (Linköping University Electronic Press, 2016). 





65. Hedström, P. & Ylikoski, P. Causal Mechanisms in the Social Sciences. Annu. Rev. Sociol. 36, 49–67 (2010). 





66. Bail, C. A. et al. Exposure to opposing views on social media can increase political polarization. Proc. Natl. Acad. Sci. 115, 9216–9221 (2018). 





67. Goldenberg, A. & Gross, J. J. Digital Emotion Contagion. Trends Cogn. Sci. 24, 316–328 (2020). 





68. Mønsted, B., Sapieżyński, P., Ferrara, E. & Lehmann, S. Evidence of complex contagion of information in social media: An experiment using Twitter bots. PloS One 12, e0184148 (2017). 





69. Del Vicario, M. et al. The spreading of misinformation online. Proc. Natl. Acad. Sci. 113, 554– 559 (2016). 





70. Hilton, S. & Hunt, K. UK newspapers’ representations of the 2009-10 outbreak of swine flu: one health scare not over-hyped by the media? J. Epidemiol. Community Health 65, 941–946 (2011). 





71. Westerman, D., Spence, P. R. & Van Der Heide, B. Social Media as Information Source: Recency of Updates and Credibility of Information. J. Comput.-Mediat. Commun. 19, 171–183 (2014). 





72. Marzouki, Y., Aldossari, F. S. & Veltri, G. A. Understanding the buffering effect of social media use on anxiety during the COVID-19 pandemic lockdown. Humanit. Soc. Sci. Commun. 8, 1–10 (2021). 





73. Lynn, C. W., Papadopoulos, L., Kahn, A. E. & Bassett, D. S. Human information processing in complex networks. Nat. Phys. 16, 965–973 (2020). 





74. Neubaum, G. & Krämer, N. C. Opinion climates in social media: Blending mass and interpersonal communication. Hum. Commun. Res. 43, 464–476 (2017). 





75. Fernández-Gracia, J., Suchecki, K., Ramasco, J. J., San Miguel, M. & Eguíluz, V. M. Is the Voter Model a Model for Voters? Phys. Rev. Lett. 112, 158701 (2014). 





76. Guilbeault, D., Becker, J. & Centola, D. Social learning and partisan bias in the interpretation of climate trends. Proc. Natl. Acad. Sci. U. S. A. 115, 9714–9719 (2018). 





77. Vosoughi, S., Roy, D. & Aral, S. The spread of true and false news online. Science 359, 1146– 1151 (2018). 





78. Brady, W. J., McLoughlin, K., Doan, T. N. & Crockett, M. J. How social learning amplifies moral outrage expression in online social networks. Sci. Adv. 7, eabe5641 (2021). 





79. Bakshy, E., Rosenn, I., Marlow, C. & Adamic, L. The role of social networks in information diffusion. in Proceedings of the 21st international conference on World Wide Web 519–528 





(Association for Computing Machinery, New York, NY, USA, 2012). doi:10.1145/2187836.2187907. 





80. Yabe, T., García Bulle Bueno, B., Frank, M. R., Pentland, A. & Moro, E. Behaviour-based dependency networks between places shape urban economic resilience. Nat. Hum. Behav. 1–11 (2024) doi:10.1038/s41562-024-02072-7. 





81. Russell, J. A. A circumplex model of affect. J. Pers. Soc. Psychol. 39, 1161–1178 (1980). 





82. Brady, W. J., Wills, J. A., Jost, J. T., Tucker, J. A. & Van Bavel, J. J. Emotion shapes the diffusion of moralized content in social networks. Proc. Natl. Acad. Sci. 114, 7313–7318 (2017). 





83. Castellano, C., Fortunato, S. & Loreto, V. Statistical physics of social dynamics. Rev. Mod. Phys. 81, 591–646 (2009). 





84. Centola, D. The Spread of Behavior in an Online Social Network Experiment. Science 329, 1194–1197 (2010). 





85. Sisson, S. A., Fan, Y. & Tanaka, M. M. Sequential Monte Carlo without likelihoods. Proc. Natl. Acad. Sci. 104, 1760–1765 (2007). 

