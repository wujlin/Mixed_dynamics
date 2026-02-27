# Fear is Polar? A Complex Network Analysis of Media Fear Appeal and Public Sentiment Transition Related to Long-COVID

# Abstract

In the era of social media, public emotions are heavily influenced by media content, which is also shaped by emotional feedback. This study investigates the dynamic interaction between media narratives and public reactions, focusing on Long-COVID discussions and changes in containment policies. Traditional methods like surveys are insufficient to examine the nonlinear emotional shifts within the complex network of interactions among stakeholders. We introduce a novel model, the Content-Sentiment Undirected Approval Graph (CSUAG), using agent-based modeling to simulate these dynamics. Additionally, we exploit Approximation Bayesian algorithms to fine-tune the model, achieving an error rate of 0.063. Our findings reveal fear significantly affects public sentiment, causing shifts between moderate and polar states, while the production of self-media content is largely driven by public emotions. This research highlights the substantial impact of Long-COVID health information on public emotions and demonstrates how self-media adjusts its risk communication in response to public sentiment. 

Key words: sentiment transition, ABM, fear appeal, Long-COVID 

# Introduction

There is considerable interest in the impacts of COVID-19 on various aspects of human society, including economic development, social justice, and mental health. While the international community has mitigated these obvious impacts, recent research has increasingly focused on patients with Long-COVID, often referred to as “Longhaulers”. We have paid attention to the discussion about Long-COVID across social media platforms, focusing on its tangible impacts—such as loss of taste and smell, memory decline, and brain fog—which resonate widely and elicit strong empathetic responses among netizens. The evolving and often ambiguous medical definitions of Long-COVID further fuel public interest as people seek clarity amid uncertainties. Before it became a focal point, patient communities on social media were already exchanging experiences and garnering support, giving rise to the term “Long-haulers” which has now achieved widespread recognition.1,2 This marks a departure from traditional agenda-setting, embracing a more inclusive communication model with diverse stakeholder involvement. Despite numerous studies on media's influence on public sentiment, there remains a gap in understanding the dynamic interaction between audience feedback and media framework,3–5 and the influence of interpersonal networks on emotional responses and media effects.6 Traditional models often overlook these complex, time-sensitive emotional transitions within networked contexts.7 To address these gaps, we propose using agent-based modeling (ABM) combined with real network data to analyze the intricate relationships between public sentiment and media narratives. ABM simulates individual and organizational behavior within networks, providing a robust tool for examining how micro-level interactions translate into broader societal patterns.8–10 This approach not only bridges micro and macro perspectives but also integrates massive datasets with theoretical insights more effectively.11 Additionally, we introduce the Content-Sentiment Undirected Approval Graph (CSUAG), a heterogeneous network model that maps strategic behaviors and information flows among diverse stakeholders, enhancing our understanding of health communication within social media platform. 

Unlike the acute phase of COVID-19 usually recover within two or three weeks, Longhaulers generally suffer persistent symptoms exceeding this duration. The identification of Long-COVID hinges on the persistence of symptoms exceeding 12 weeks after initial infection, showcasing a range of symptoms including but not limited to fatigue, loss of taste or smell, memory impairment, challenges in concentration, and 

cardiovascular issues, without any other medical rationale.2 Yet, the definition of Long-COVID provokes debate among researchers, with discussions pointing to a variety of potential durations for the ailment's endurance.12–14 Additionally, much sharing symptoms with other conditions confound the direct diagnosis of Long-COVID.15 Such ambiguity brings about inherent contradictions within the discourse on Long-COVID, laying the base for numerous societal dilemmas.13 A major current focus is on the various aspects that Long-COVID influences. The escalation of societal risks and uncertainties leads to misinformation, opinion polarization,14stigmatization,13,16,17and deteriorating socio-economic conditions,12 disproportionately impacting vulnerable demographics, notably in developing regions.14 Beyond its macrosocial impacts, studies highlight the heavy anxiety and frustration derived from enduring physical and psychological unease, further intensified by the ambiguity of the disease's future trajectory and insufficient medical care.18 These adverse emotional states are associated with increased suicidality, adversely affecting the well-being of individuals.19 

It is generally accepted that social media platforms have become essential spaces for ordinary people to express opinions and emotion. As for public sentiment, much studies have paid attention to its attribute of evolution. Nabi et al. have employed the concept of “emotion flow” to elucidate the progression of emotions experienced by audiences throughout their exposure to a health message. The theory of emotion flow argues that messages, such as fear appeals, can sequentially evoke a variety of emotions in viewers, 

with emotional states evolving as the content of the message changes.20,21 In summary, 

Nabi posits that users' emotional states are not static but rather evolve during the exposure to health information. The pivotal determinants of this emotional evolution are the audience's sensitivity to and receptivity towards persuasive messages. Numerous studies have claimed that sentiment can be classified into different types. Changes in emotion can prompt individuals to adopt different behavior patterns. The Motivated Attention theory suggests that fear can make people avoid situations perceived as threatening, leading them to disregard related messages. Although fear can initially cause avoidance, the desire to protect oneself may still drive information processing, especially if it could reduce uncertainty.22 Berger noted that uncertainty increases with unpredictable events, but can be mitigated by shared beliefs and information networks.23 Building on this, Anxiety Management theory focuses on how 

emotions like anxiety influence behavior. He proposed that extreme fear would prevent 

individuals from focusing on information and prompt passive or avoidant responses to crises. Conversely, too little fear results in low engagement, where individuals process information superficially.24 Thus, both very high and very low emotional arousal can impair effective communication and information dissemination. Conversely, moderate levels of anxiety and tension can enhance the ability to navigate complex informational environments. In this study, we classify extreme emotional arousal, whether high or low, as a “polar” state. In contrast, we define the middle level of arousal as a “moderate” emotional state. 

Given the remarkable impacts on human behaviors, the emotion transition has generated considerable recent research interest. Most of them put forward and examine possible hypothesis to explain the reason why emotion transition happened. One factor worthwhile to consider is the risk information various medias communicate on the social network, which results in more stress and fear. Researchers have studied many aspects of public sentiment on social media. A vast spectrum of studies on public sentiment transition has centered on sentiment analysis across extensive datasets, scrutinizing the dynamic shifts in public emotions aligned with pivotal events.25–29 Scholars have also exploited topic modeling to analyze the issues that different emotional states prioritize.30–32 Furthermore, survey data and statistical examination has been employed to explore factors influencing emotional shifts, such as personal income, education level, content themes, and risk perception.33,34 The interplay of emotions serves as a fundamental perspective for unraveling the social issues during the pandemic, shedding light on the multifaceted behaviors such as the pursuit of social support,35,36 rumor propagation,37–40 polarization,41,42 and cyber bullying.43 During the pandemic, social media has been essential for accessing information and support network, with increasing research underscoring the critical role of emotions in these interactions. Liu highlighted the significance of information-seeking behaviors on social media as a catalyst for containment measures, significantly mediated by sentiment such as low-arousal sentiment, worry.44 Concurrently, Neely elucidated the critical role and potential pitfalls of social media in health information spread, alarming the psychological ramifications of information overload and heightened anxiety.45 Given the ambiguous diagnostic criteria for Long-COVID, the resulting social dilemmas require a nuanced, emotion-centric analytical approach.18 Since the profound integration of emotions in social media interactions, exploring the public sentiment transition worths deeper investigation.46 The public discourse frequently revolves around perceived threats. Particularly during significant public crises, there is a pressing need for the public to access ample information to manage the prevailing uncertainty. However, the vast and complex landscape of information often triggers public panic, a situation that becomes particularly acute in the age of social media, where the traditional gatekeeping functions of the mass media are weakened.47 Evidence shows that social media posts can also generate public fear emotion.48 As suggested by the sociological concept of “Risk Society”,49 the notion of risk increasingly occupies a pivotal role in public discourse, with public perceptions of risk contributing to shifts in societal emotions and psychological states, particularly in terms of negative emotions.50 Negative sentiments, in comparison to their positive counterparts, attract more attention,51 with anger and fear particularly influential in driving social mobilization and fueling rumor dissemination.25,40 The spectrum of negative emotions needs a nuanced analysis; varying degrees of emotional arousal impact how individuals perceive their surroundings and make decisions. Sentiments of high arousal, such as anger, intensify focus, notably when processing external evaluations or confronting stigmatizing narratives.52 Additionally, varying media content can induce different levels of arousal.53–55 The degree of sentimental arousal 

plays a pivotal role in shaping behavioral responses. While fear and anxiety might catalyze protective measures to mitigate risks, boredom tends to weaken the drive for engaging in preventative actions.52,56 Low-arousal negative emotions like boredom, for instance, exert distinct impacts, such as distorting the perception of time.57 

This paper focuses on the dynamics of public negative emotions, aiming to decode the contagion of emotions by analyzing how individual information processing and exchange patterns contribute to their emotion transition. Drawing inspiration from the Shannon-Weaver model, we introduce a novel framework to analyze the discourse surrounding Long-COVID on Chinese social media platforms. Shannon and Weaver's model demonstrates how information transmit from its origin to its recipient.58 Further research has refined this model, with DeFleur arguing that information diffusion is inherently a two-way process. Here, feedback from the receiver prompts the source to refine its communication approach, thus enhancing the likelihood of message homogeneity.59 Given individuals' tendency to form connections based on shared knowledge and interests, homophily emerges as a pivotal element in the fabric of social and informational networks.60–62 On social media, if a user frequently likes or reposts another user's posts, it may indicate a shared perspective on a topic.63 In East Asia, the government pays significant attention to homophily in the information propagation process among subsystems within the social system, aiming to achieve consensus among government and societal sectors on national development goals and major public event decisions.64 In China's public life, the primary mode of information flow and diffusion remains top-down, driven by the government as the principal information source.65,66 However, social media has enabled voices beyond traditional institutions, creating a diverse landscape on Chinese platforms where various stakeholders share information to pursue their interests.67,68 In the context of public health concerns like Long-COVID, the variance in stakeholders' perceptions catalyzes the flow of information. From the audience's perspective, Festinger and Newcomb, among others, have discussed how perceptual dissonance can induce psychological stress, driving individuals towards information seeking from media or alternative sources.69,70 McGuire further explained that these discrepancies allow political groups to influence public perceptions either by directly participating in events or by manipulating media narratives.71,72 This interactive model highlights that distinct stakeholders display unique communication behaviors, necessitating a novel framework that encompasses these complex interactions. In the Chinese social media ecosystem, the main information disseminators are mainstream media, We Media, and ordinary people. We identify these stakeholders by analyzing keywords related to industry, workplace, and profile descriptions (Supplementary Section 6). 

To explore the interrelationships among different stakeholders, we propose several critical hypotheses as follows: 

Social Stability. Mainstream media, including television, newspapers, radio, and leading online platforms, exerts significant influence and accessibility across society. 

These outlets are highly credible and authoritative, reaching a wide audience. They are crucial in shaping the public agenda, disseminating information, and influencing opinions and behaviors. Shaw indicated that mainstream media often initiate the flow of information, playing a key role in the propagation chain. 73 According to Lasswell, mass media enable environmental surveillance, keeping individuals informed about global events.74 Beyond surveillance, they coordinate societal responses to events, guiding public interpretations and norm setting.75,76 

In China, mainstream media are often viewed as government extensions, necessitating a focus on public interest and political stance in their operations.77 When public discussions arise on topics like Long-COVID, mainstream media guide opinions. Barbieri and colleagues argue that media establish authority through content production. 78 On social media, mainstream outlets extend their influence by disseminating explanations and guidelines about Long-COVID through microblogging, significantly shaping public understanding and responses. As their primary aim is to maintain social order,79 mainstream media also monitor public feedback, passively adjusting their content to maintain a balance that aligns with societal expectations and norms. 

Popularity. We Media typically involve content creation and sharing by individuals or small groups through online platforms. Unlike mainstream media, We Media adopt a decentralized and customized approach, quickly adapting to and reflecting audience interests to foster more immediate and engaging interactions. It plays a crucial role in disseminating information, effectively bridging societal issues and the general public. In contrast to mainstream media, We Media, driven by market profitability, increasingly cater to the psychological tendencies and preferences of its audience.80 In the competitive media market, where audience attention is limited, We Media frequently prioritize more sensational content, sometimes spreading misinformation to boost 

network traffic.81 As explained by van Dijck, social media operates under its own logic, with programmability, popularity, connectivity, and datafication shaping content circulation. 82 Platform algorithms often favor content that triggers significant interaction and engagement. Research highlights that high-arousal negative emotions, such as anger, tend to stimulate more discussion among netizens than other emotions. 83–85 Thus, We Media often emphasize such sentiments. In China, mainstream media still set agendas and define political boundaries. 86 In order to attain legitimacy, We Media must align with mainstream media while also catering to the needs of their audience by creating content that resonates with or sparks audience interest, fostering connectivity, and customizing content which reflect audience preferences. 

Fear appeals. Highlighting risks and uncertainties within a message can alert individuals to current threats and motivate actions to mitigate future risks. Extensive research in health communication has demonstrated that disease-focused fear appeals can evoke a range of emotions, including anger, disgust, and sadness.48 Within the 

CSUAG network, users receive information from media in their news feeds, eliciting emotional and behavioral responses to content related to Long-COVID risks.87 

Besides fear appeal information, an individual's emotional state is shaped by the emotional contagion from fellow users, with the psychology of convergence encouraging the maintenance of homophily.88 Some studies have shown that individuals tend to engage selectively with others, preferring to establish connections with those who are more similar to themselves, a behavior that enhances homophily among individuals.89,90 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/e61834da8844a23d7aaf917c6f2fe0da190ba6f54898e29752c5c15c51a5609d.jpg)



Fig. 1| Content-Sentiment Undirected Approval Graph (CSUAG). The yellow layer represents state-controlled media, the green layer depicts We Media ecology, and the red layer signifies the ordinary people's network. Solid arrows illustrate the pathways of influence, while dashed lines denote the connections between media entities and individuals.


As depicted in Fig. 1, stakeholders with different motives participate in the information diffusion process. This graph is segmented into three distinct set of nodes: the mainstream media, M ; followed by the We Media,W . Both sets utilize content as the primary means of exerting social influence. In contrast, the ordinary people constitute the other node set S, where people sentiments are treated as valuable informational 

inputs. These sentiments feed back into the media ecosystem, affecting content creation and dissemination from media. To explore connections among these nodes, our study exploits repost behavior data, a common method in social media that significantly facilitates information dissemination. Unlike following or commenting, repost more clearly indicates endorsement of specific viewpoints and more distinctly delineates the pathways of influence.63,91 Within the CSUAG, every edge symbolizes a potential influence from one node to another, with these edges diverging based on the distinct motivations of each stakeholder.63 In the information flow about Long-COVID, statecontrolled media adopts a more neutral content approach to guide ordinary people in keeping societal emotions within an appropriate range, while the intermediate We Media layer considers both mainstream content tendencies and partial audience emotional changes. Ordinary people, positioned at the end of the propagation chain, are influenced by media information, and emotions as feedback information prompt media to adjust their content production. Our research performs the ABM experiment based on the dynamic mechanisms presented in Fig. 1, with stakeholders' states changing according to the information received. 

# Results

As mentioned earlier, our purpose is to refine the analytical framework for health information dissemination using ABM. This model integrates a range of agents— ordinary people, mainstream media, and We Media—each defined by attributes derived from empirical data. These agents are connected by forwarding acts, which represent approval and facilitate the flow of information. Our ABM aims to uncover the interplay between public emotions and media content by simulating the decision-making processes of these agents based on the received information. This approach promises to deepen our understanding of the dynamic relationship between media content related to health risks and public emotional responses. We conducted our simulations on the proposed network, CSUAG (including 10370 nodes), along with the empirical data calibration. Leveraging these ABM experiments, we can simulate the communication process between different stakeholders on the social media involved Mainstream Media, self-media and public. Furthermore, utilizing simulation method provide us more direct access to estimate the causal effect because of its accurate control over experiment parameters. ABM essentially views the social phenomenon as stochastic situations, and parameter calibration make our model prediction results closer to the empirical data. Most importantly, these experiments enable us observe how the theoretical rules bring about the outcome, which reveals the Longitudinal causal mechanisms and validate the proposed hypothesis vividly.92,93 

From the emotion flow point of view, the content module is based on the approval behavior that connects between media outlets and individuals. Following the emotion transition among the netizens, the media would adjust their viewpoints in terms of social hot spots. As for the Long-COVID, the risk signals released by media shift 

together with the public sentiment alteration. To simulate the risk propagation occurs on the media nodes, the ABM implements a stochastic, discrete-time information transmission model on the interaction network with media nodes transitioning between risk states based on the different impact rules. Specifically, the state-controlled media consider the social stability on the top of list and self-employed media focus on the network traffic. 

Given exterior information stimulus especially the fear appeals, individuals need to handle these risk signals. Too much alarms might leave their sentiment out of control. In particular, serious syndromes related to Long-COVID easily make people obsessed in the bad emotion such as depression, helpless and so on. Therefore, the sentiment module takes the risk information as input and changes the states dynamically. The ABM will operate the content module and sentiment module at the same time. 

We calibrate the model’s key parameters, including the parameter controlling the states changes, to fit the actual statistics obtained from empirical data. ABM experiments begin on 14 October 2022, end on 18 January 2023. We also consider the containment policy transition effect. The important policy time points we selected are 14 November 2022 and 8 December 2022(Supplementary Section 4). 

# Sentiment validation

In our sentiment analysis, the calibrated model convincingly reproduces empirical observations, capturing the rise in moderate sentiment immediately following the policy implementation on November 14th, as illustrated in Fig. 2. This trend continues with a gradual increase in sentiment post-December 6th, consistent with findings from previous studies that noted a mild improvement in public mood following lockdown lifts.94,95 The model’s robustness is underscored by a significant Pearson correlation coefficient of 0.84 (P value: 0.000147). However, as depicted in Fig. 1, an outlier on November 28th deviates from the expected trend, indicating external influences on sentiment dynamics. This anomaly aligns with intense discussions about relaxing China's dynamic-zero COVID-19 policy, a period characterized by heightened uncertainty and extensive public discourse. Similar to the results obtained by Király et al., this phase amplified fears of illness, financial insecurity, and uncertainty about the future, exacerbating stress, anxiety, and depression. The resulting spike in polarized sentiments highlights the complex relationship between public sentiment and health policy shifts.96 These insights offer a nuanced view of how policy changes impact public emotions. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/f424598b307521c342b0defa3c7bce08cda260f404381e39c75a0423cd42d1b9.jpg)



Fig. 2| Temporal Dynamics of Moderate Sentiment. The figure illustrates the evolution of the moderate sentiment proportion, with model simulations against empirical observations. The continuous line denotes the modeled sentiment trajectory, while the grey markers represent empirical data points aggregated on a weekly basis.


# Media module validation

Our model demonstrates a strong correlation with observed risk perceptions, evidenced by a Pearson correlation coefficient of 0.99 (P value $7 . 1 5 \times 1 0 ^ { - 6 } )$ ), indicating excellent alignment with empirical data as shown in Figure 3a. During this period, a more transmissible Omicron sub-variant breached the barriers established by the zero-COVID strategy, prompting the Chinese government to announce an easing of these policies.97 Faced with escalating public stress, the government effectively used statecontrolled media to influence public sentiment. As a result, mainstream media began to question the scientific associations between Long-COVID and symptoms like memory loss, chest pain, and ankle pain, thereby alleviating public concerns about these risks. 

Furthermore, our comprehensive analysis of business strategies within self-media ecosystems reveals that these platforms proactively track and amplify societal trends to attract followers, often emphasizing the potential risks associated with Long-COVID (Fig. 3b). However, as government policies gradually relaxed epidemic controls, these platforms aligned with mainstream narratives, downplaying Long-COVID risks. Compare well with prior research, our findings also highlight the evolving relationship between official media and self-media during the middle and later stages of the COVID-19 pandemic. The government and official media intensified efforts to monitor and regulate the influence of self-media on public emotions and behavior, improving the oversight of online information and official media account operations. This strategy 

aimed to ensure consistent and coordinated information dissemination between official and self-media to prevent misinformation.98 On the microblog platform, governments and mainstream media predominantly influence topic propagation and sentiment contagion, underscoring their pivotal role in guiding public opinion. Establishing a robust information disclosure mechanism for public events is thus essential.91 

Lastly, our We Media risk model, assessed using a Pearson correlation coefficient of 0.48 (P value of 0.08), shows reasonable alignment with empirical data. However, it is noteworthy that outliers, such as the low-risk instance observed on October 28, suggest the model's sensitivity to fluctuations in public discourse during uncertain policy transitions. This anomaly likely reflects the public's reaction to these changes. 


a



b


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/9aa564be96efa8c0be2e0fff8395192ebee2ae172864768df870a502a0606c46.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/37d335d75b2e26604531743bdcb0c8d31c9c8acd7c8f3ffcaa360958c8ca511c.jpg)



Fig. 3| Evolution of Long-COVID Risk Perception in Media Coverage. Panels a and b contrast model simulations with empirical data on the portrayal of Long-COVID risk. The solid line delineates the trend in risk perception, with grey markers denoting weekly-aggregated empirical observations. Panel a represents the percentage of state-controlled media acknowledging the danger of Long-COVID. Panel b quantifies the disclosure rate of Long-COVID risks by independent media entities.


# The Co-evolution of Fear Appeal and Public Sentiment

As shown in Fig. 4, we analyze the states of different groups at three distinct time points. Initially, as depicted in Fig. 4a, the majority of state-controlled media $( 9 8 . 8 \% )$ and self-media $( 9 6 \% )$ described Long-COVID as a significant health risk. Concurrently, the general public's emotional state was unstable in this informational 

context. In early October 2022, both the Chinese government and media maintained a cautious stance towards Long-COVID. For instance, on October 14, 2022, experts from the Center for Disease Control and Prevention emphasized through media outlets that Long-COVID could have a wide range of symptoms persisting for weeks, months, or longer. 

In contrast, by the time represented in Fig. 1b, media portrayal of the risks associated with Long-COVID had diminished significantly. Mainstream media completely ceased highlighting these risks, while $6 8 . 1 \%$ of We Media continued to acknowledge the harm caused by Long-COVID, although this was a decrease of approximately $1 8 \%$ from the initial phase. This period saw moderate sentiments becoming more prevalent, a result of the interplay between public pressure and government crisis management. As discussed in the introduction, mainstream media adjusted their reporting frameworks on Long-COVID in response to shifts in public sentiment, influencing audience risk perception. Between December 6 and 16, 2022, the government progressively relaxed its zero-COVID policies, reflecting what Willson identified as a response to public emotional instability and a crisis of trust in government, leading to a U-turn in containment policies.97 

The outcomes of the final simulation step are presented in Fig. 4c. Although most mainstream media continued to deny the scientific basis of Long-COVID, $9 6 . 3 \%$ of We Media persisted in asserting its existence and danger. The proportion of the population in a moderate mood state decreased slightly by $2 . 9 \%$ from the previous time point. By early 2023, China had largely ended its strict pandemic controls. On January 8, 2023, the COVID-19 risk level was downgraded, leading mainstream media to halt their fearmongering regarding Long-COVID. However, the complete relaxation of containment policies also led to an increase in infection rates, heightening public anxiety about Long-COVID, with Self-media quickly responding to and amplifying this public stress. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/886324dc4cdf76f4bb174375ade0fe1a5038ead22e745b7f446a3a245cfde0ca.jpg)



Fig. 4| Transition Dynamics of Risk Signals and Sentiment Over Time. a, Represents the initial state of the CSUAG model at $\scriptstyle { \mathrm { { t } } = 0 }$ . b, Describes the intermediate phase at $\scriptstyle \mathbf { t = 6 2 }$ , chosen due to its association with the final stages of governmental policy transition, where the last batch of policies was enacted. c, Depicts the final simulation step at $\scriptstyle \mathbf { t } = 9 1$ . Each solid point is color-coded to indicate


the state at a specific time. Simulation results are based on the optimal parameter combination （ $\alpha { = } 2 . 9 8 7 5$ , β=18.7623, $\theta { = } 0 . 1 7 5 3$ , $\delta _ { \scriptscriptstyle 1 } = - 0 . 1 5 3 2$ , $\delta _ { 2 } { = } 0 . 1 0 2 6$ , $\delta _ { 3 } \mathrm { = } \mathrm { - } 0 . 1 3 3 0$ , δ 4=0.1507,  γ 1=−0.9291,  γ 2=0.1293,  γ 3=−0.3358,  γ 4=0.1126） . Visualizations were $\delta _ { 4 } = 0 . 1 5 0 7 ^ { \prime } ~ \gamma _ { 1 } = - 0 . 9 2 9 1 ^ { \prime } ~ \gamma _ { 2 } = 0 . 1 2 9 3 ^ { \prime } ~ \gamma _ { 3 } = - 0 . 3 3 5 8 ^ { \prime } ~ \gamma _ { 4 } = 0 . 1 1 2 6 ^ { \prime }$ 

created using Arena3Dweb99, which imposes limits on the number of network nodes; therefore, a random sample of $1 0 \%$ of nodes was used for visualization. 

# The coupling relationship between Media and Ordinary People

Through extensive experimentation with numerous parameter combinations, we ultimately selected 909 combinations with errors less than 0.07. The optimal parameter set achieved a minimal error of 0.063. Figure 4 illustrates the distribution of three key parameters within these combinations, interpreted under Approximate Bayesian Computation as the posterior distribution of parameters, obviating the need for direct likelihood evaluation. Notably, the distribution of specific parameters, such as the Popularity rule $( \beta )$ and Fear appeal (θ), exhibited pronounced peaks. 

As can be seen in Fig. 4, the parameter $\beta$ demonstrates how We Media dynamically adjust their representation of Long-COVID risks in tune with the emotional feedback of the audience within specific periods. This behavior, which aligns content with audience sentiments, has been supported by studies in social media dynamics. Given social media platform control over traffic, these platforms use algorithms to prioritize content that resonates emotionally with users, effectively curating feeds to match user interests and emotional tendencies.100–103 Consequently, the emotional content that garners engagement influences social media content strategies, making user emotions and opinions pivotal in guiding content creation. 

In most cases, self-media, with its accessibility, low barriers to entry, immediacy, decentralization, and rapid dissemination, has revolutionized traditional media's approach to production, distribution, and control. These platforms have become the primary sources for public access and dissemination of Long-COVID information. Notably, individual and niche Weibo accounts, have taken on critical roles in risk warning, information dissemination, and guiding public opinion, effectively acting as key “risk perceivers”. As shown in Fig. 4, the fear or risk appeal factor, θ, highlights the substantial impact of media-driven risk signals on users' emotional state transitions, supporting existing research on the persuasive power of health and medical news risk information. This observation is consistent with Lang’s Limited Capacity Model of Motivated Mediated Message Processing, which posits that emotional changes affect cognitive resource allocation, thereby influencing the encoding, storage, and retrieval of message elements.104 And can spur widespread social movements depending on the elicited emotions' conduciveness to health campaign objectives.20 As Pedrosa et al. concluded, information overload and indistinguishable outbreak content heighten public anxiety.105 This correlation is also illustrated in Fig. 2 and Fig. 3b, where the 

fluctuations of the two curves largely coincide. For example, around November 14th, as self-media reduced the dissemination of risk signals about Long-COVID, there was a corresponding increase in the public's moderate emotional response. In mid-December, following the relaxation of containment policies, a short-term rise in death and infection rates heightened public panic about Long-COVID, coinciding with an increase in risk signals from self-media about Long-COVID. 

Conversely, our model generated a flat posterior for the Social Stability factor $( \alpha )$ , indicating an inability to precisely identify parameter values, likely due to limited empirical sensitivity. This challenge may arise from shortcomings in traditional propagation management methods. As outlined in the introduction, the Chinese government and official media have utilized a top-down approach to manage and disseminate risk information. Aligning with previous studies, our findings suggest that during the pandemic, official media failed to establish effective communication channels between the government and the public, leading to sluggish responses in managing public opinion and controlling rumors online.98 This issue may be exacerbated by data deficiencies, such as an insufficient sample size in the Mainstream Media dataset. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/8bf4f8265e5d93c7da2d37fe438fb7a6add64f576beb6f76d2ec2027a3a22a37.jpg)



Fig. 5| Posterior of content module and sentiment module parameters. These histograms show the full range of sampled parameters on the horizontal axis. One can interpret the priors as flat over these ranges.


# Discussion

In a complex external information environment, public emotions are dynamic, evolving through experiences of exposure to health messages. Recent research indicates that 

patients infected with the novel coronavirus often experience lingering symptoms, prompting the formation of discussion communities on social media. These groups have attracted attention from mainstream media and independent content creators, bringing Long-COVID discussions into the public spotlight. The portrayal of Long-COVID as a condition is a collective construction shaped by public discourse, media, and other stakeholders, undergoing an evolutionary process. Media plays a dual role: shaping the perceived threat of Long-COVID, which influences public risk perception and emotional responses, and receiving feedback on public emotional states, which is processed differently by mainstream and independent outlets due to varying motivations and perspectives. To understand how Long-COVID risk information affects public emotions, our study employs Agent-Based Modeling (ABM). This approach designs micro-level interactions from the perspectives of various stakeholders and calibrates model parameters with empirical data to enhance the model’s explanatory power. 

Our research establishes a correlation between public sentiment and media portrayal of Long-COVID. The depiction of Long-COVID as a significant health risk influences shifts toward more polarized emotional responses in the public. Social media users encounter a wide array of information from both mainstream and independent media sources. Our model, utilizing an Approximate Bayesian calibration algorithm, shows that independent media's fear appeal has a stronger impact on public emotions, likely due to its diverse audience. In contrast, mainstream media, with its more uniform content and style, engages less interactively with its audience, resulting in a more subdued influence. Public emotion evolution critically guides media content production. Mainstream media uses public sentiment as a barometer to steer public opinion, tailor news dissemination, clarify misinformation, and stabilize social order by responding to shifts in public emotions. Independent media view public sentiment as a key to generating platform traffic. They selectively publish content that resonates with public opinions and emotions to attract clicks and engagement. Our model also accounts for the feedback effect of public emotions, confirming that public emotional responses shape media content production. Additionally, we consider the influence of government policy. By incorporating significant policy announcements as nonlinear expressions in our models, we enhance the explanatory power of the simulations. These results indicate that government actions, as exogenous factors, can significantly influence short-term variations in risk information and public emotional responses. 

Our study acknowledges typical limitations inherent in modeling complex social interactions. Primarily, our model exclusively focuses on the dynamics between individuals and media, omitting other influential relationships such as community bonds and interactions among internet strangers, which can significantly affect an individual's emotional state. Despite these omissions, we believe our core findings remain valid for elucidating short-term interaction dynamics between individuals and media. A significant limitation is potential sampling errors due to the sensitive nature of discourse around Long-COVID, where some comments may be censored or hidden, 

leading to sample bias and impacting the external validity of our model. Although this limitation is intrinsic to our analysis, we believe the volume of our dataset is sufficient to generalize our findings. Additionally, our model does not account for the influence of emergent events or social campaigns, which could limit its explanatory power during specific periods. This gap highlights the need for future research to more accurately capture the nuances of dynamics over short time spans. Another constraint is our equal influence weight assignment to each media outlet in our CSUAG, effectively creating an equal-weight network. This method does not accurately reflect the actual dynamics of information propagation. Future work will aim to incorporate metrics such as click rates, likes, and shares to differentiate the influence weights of various media outlets, aligning more closely with real-world propagation scenarios. 

# Methods

In this section, we outline the components of the ESUAG model, including its network components, the dynamics among different agents, and the calibration of model parameters. Our study integrates semantic sentiment analysis to gather empirical data, capturing both the risk-related content shared by media and public sentiment, sourced from the Weibo platform. Utilizing these materials, we categorize data from media and individuals to ascertain the interactions between various entities, forming the foundation of the CSUAG. Additionally, we employ semantic lexicon technology to extract latent risk information from media content and sentiment scores from individual texts. Our dynamic model, based on the formulated hypotheses, undergoes calibration in each iteration against empirical data, employing Bayesian approximation to refine parameter combinations. To minimize estimation errors, empirical case data was used as both the input-output reference for the model, enabling the testing of various rule sets. Incorporating real-world data at different stages of model development ensures that a theoretically derived model can accurately reconstruct and potentially forecast empirical patterns. 

The primary aim of these experiments is to investigate the interactions and evolving relationships among various stakeholders in the discourse surrounding Long-COVID. 

# Data and Code availability

In this study, we analyzed public discourse on Long-COVID from the Sina Weibo platform, one of China's most renowned and popular social media platforms. To gather data, we targeted specific topic entries to isolate relevant texts, ultimately obtaining 65,395 posts and 69,628 retweet relationship datasets. We crawled microblog data, which included original posts, retweets, comments, the number of retweets, forwarding links among users, and user profiles. We manually filtered out irrelevant posts to reduce data noise. 

To increase data density, we selected a concentrated time period (14 October 2022 - 18 January 2023) that coincided with a pivotal transformation in China’s containment policy and focused explanations on Long-COVID by health authorities (Supplementary Section 5). We assumed public sentiment transition was particularly marked during this period, making it suitable for model calibration. Moreover, to enhance randomness, we conducted random sampling of the data from this period, a method previously described by Pangallo et al.55 Using comprehensive data from this timeframe directly in simulations would treat our estimates as initial conditions, thereby diminishing the model’s explanatory power. 

The data and code underpinning all quantitative results in this study are publicly accessible on GitHub. Due to privacy concerns, some portions of the initial dataset that contain personal information are not available for public release. 

# Semantic sentiment analysis

To assess the sentiment status of ordinary people, we employed a semantic lexicon for detailed classification work. Given the extended version based on DLUT-Emotionontology,106 we compute the sentiment score for each emotion type. In this process, we took into account the subsidiary effect of the negative words. According to linguistic research, the negative words usually form the adverbial-center structure or attributive-center structure. Therefore, with the aid of dependency parsing, we extracted the sentimental relation pairs such as (not, happy) using Language Technology Platform toolkit developed by Harbin University of Technology. 

If the sentimental word w is modified by negative words, its emotional intensity changes as shown in equation (1). 

$$
E (\text {P a i r}) = \left(- 1 \frac {\binom {n} {2}}{2} \sqrt {E (w)}\right) \tag {1}
$$

In the equation, $E ( { \mathrm { P a i r } } )$ denotes the new intensity of the emotional word w modified by negative words, $n$ denotes the number of negative words that modify the emotional word $w$ , and $E ( w )$ denotes the initial intensity of the emotional word w. Given that negative words can change the type of emotional words, we carried out negative conversion to emotional words. In this study, we have three emotion types. The emotional type of each microblog post is determined by the emotional type of which the sum of the intensity values of the emotional words is larger than that of other emotional types in the microblog post as shown in equation (2). 

$$
\text {S e n t i m e n t} \quad \text {i}, \text {T y p e} \left(\max  \left(\sum_ {j = 1} ^ {n} w _ {i j}\right)\right) \tag {2}
$$

In the equation, $w _ { i j }$ is the intensity value of the $j$ -th emotional word belonging to the i-th emotional type in the microblog post, and “Sentiment type” is the emotional type of the microblog post. The emotional intensity of the microblog post equals the average intensity of the emotional words belonging to the emotional type in question. The emotional tendency of a user in a certain period is determined by the emotional tendency of the microblog posts generated by the user in that period. 

Considering the risk degree of media content, various methods have been previously utilized to gauge the implicit information within texts, such as biases107 and risks50. In this study, we have employed semantic analysis, leveraging extended risk assessment lexicons. Likewise, we calculate the risk score for each type of risk and assign the type with the highest score as the state value for the media content. 

# Network

Consider three sets of nodes: S, M , and W , each comprising nodes in distinct states. The set S, representing ordinary people, is defined by its adjacency matrix $S _ { i , j }$ . This matrix $S _ { i , j }$ delineates the interaction directed from individual $j$ towards individual i. Let $M _ { u } = m _ { 1 } , m _ { 2 } , \ldots , m _ { m _ { m } }$ denote the set of nodes corresponding to content from Mainstream Media, with $A _ { u , i }$ representing the set of edges connecting individual $i$ to content $u$ . In a similar vein, $W _ { v }$ denotes content from We Media, and the matrix $B _ { v , i }$ encapsulates the relationship between user i and content v. Additionally, we introduce three vectors to depict the status of nodes at time  t: the vector $L _ { t } ^ { M } { = } \backslash \{ L _ { t } ^ { m _ { i } } | L _ { t } ^ { m _ { i } } \in \backslash \{ 0 , 1 \backslash \} , m _ { i } \in M \backslash \}$ 

characterizes the degree of risk associated with Long-COVID as implied by Mainstream Media content, while a similar vector W pertains to We Media. The value $L _ { t } ^ { W }$ Lt 

of zero indicates that the media-reported information suggests no significant symptoms associated with Long-COVID. The last vector, LS=\{ Lsi ∣ Lsi ∈ \{ 0 , 1 \} , s ∈ S \}, $L _ { t } ^ { S } \mathbf { = } \backslash \{ L _ { t } ^ { s _ { i } } | L _ { t } ^ { s _ { i } } \in \backslash \{ 0 , 1 \backslash \} , s _ { i } \in S \backslash \} ^ { } ,$ 

indicates the sentiment status of individual $i$ , where $L ^ { s _ { i } } = 1$ signifies polar sentiment, and $L ^ { s _ { i } } { = } 0$ represents a moderate arousal state. 

# Content Module

# 1. Mainstream content dynamics

In the context of moderating public sentiment, state-controlled media are posited to favor the dissemination of content that could foster a more balanced sentiment distribution among the public. To denote the role of public sentiment, we consider the sentiment disparity, $d = n _ { p o l a r } - n _ { m o d e r a t e }$ , with $\backslash \{ n _ { p o l a r } , n _ { m o d e r a t e } \} \big \}$ quantifying the populations of polar and moderate sentiment expressions. 

We then propose the probability function $P _ { u } ,$ that governs the likelihood of a shift in the content's perceived risk level. This model is delineated as follows: 

For a positive sentiment disparity $( d { > } 0 )$ , the probability $P _ { u }$ that Mainstream Media content will take transition determined by the relative magnitudes of $n _ { p o l a r }$ . Specifically, the function is articulated as equation (3): 

$$
P _ {u} = 1 - e ^ {- \alpha d} \tag {3}
$$

Considering the staying harmonious is the most important logic adopted by the statecontrolled media, we assume the media will tend to release the none-risk news when the public sentiment out of control. 

# 2. We Media content dynamics

Given the popularity logic We Media holds, We Media platforms, recognizing the appeal of moderation for its tranquility and stability, are predisposed to tweak their content to incite audience engagement. The likelihood of any content shift within a We Media platform, $P _ { v } ,$ , is then uniformly modeled to account for audience sentiment distribution, irrespective of the initial risk portrayal. Specifically, the probability is formulated as equation (4): 

$$
P _ {v} = P _ {N R \rightarrow R} = \frac {1}{1 + e ^ {- \beta \cdot \left(p _ {\text {p o l a r}} + r _ {m}\right)}} + \pi_ {r} \tag {4}
$$

where $p _ { p o l a r }$ denotes the proportion of individuals classified as polar negative emotion, $r _ { m }$ serves as the percentage that state-owned media hold Long-COVID is highly dangerous and $\pi _ { r }$ is the government effect (Supplementary Section 4). 

# Sentiment Module

To elucidate the dynamics of emotional responses, we adopt a probabilistic function and a threshold model to analyze sentiment transitions under various conditions: 

(i) Fear Appeals: Defining $r = n _ { r i s k } - n _ { n o r i s k }$ , we consider the probability of an individual's sentiment status transitioning at time $t$ , denoted as $g ( i , t )$ . Also, we include the government effect $( \pi _ { s } )$ in the Equations. The transition probabilities are expressed as $P ( M o d e r a t e  P o l a r )$ and $P \mathcal { ( P o l a r  M o d e r a t e ) }$ . 

The emotion polarization probability is defined as equation (5): 

$$
g (i, t) = P \left(M o d e r a t e \rightarrow P o l a r\right) = \frac {1}{1 + e ^ {- \theta r ^ {2}}} + \pi_ {s} \tag {5}
$$

Else the probability of emotion convergence becomes equation (6): 

$$
g (i, t) = P \left(P o l a r \rightarrow M o d e r a t e\right) = \frac {e ^ {- \theta r ^ {2}}}{1 + e ^ {- \theta r ^ {2}}} + \pi_ {s} \tag {6}
$$

(ii) Homophily: Define $\eta ( i ) ^ { t }$ as the proportion of neighbors within $S$ sharing the majority sentiment at time t . 

Furthermore, let $\sigma ( i )$ represent the sensitivity threshold for social influence. If the η ( i )t−1 > σ $\begin{array} { r } { \eta ( i ) ^ { t - 1 } \sigma ( i ) ^ { , } } \end{array}$ an individual will choose to align their sentiment with the majority's state. 

Incorporating these two principles, we postulate that a sentiment shift occurs if either condition is met, facilitating an understanding of how emotional dynamics and social influence drive sentiment changes within networks. 

# Calibration

To make our model align with empirical data better, we utilize Approximate Bayesian Computation (ABC) algorithm which find the best parameters based on prior distribution and numerous iterations. In our model, we need to estimate eleven main parameters. For the prior parameter distribution, we identify the first promising region by running preliminary simulations. Then, we randomly sample 10000 parameter combinations and input them into the model. For each parameter combination, we use 10 random seeds to run model 10 times, as this is known to reduce the estimation error. We calculate the mean error between model results and empirical data and select the parameter combinations that result with error no more than 0.07. 

The methodology steps are detailed in Fig. 6. 

Collected Dataset 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-27/c7ce1fb9-f693-4a87-9f99-35094107b4bc/a2e9d3e687cc0880a203d1bca33918189b6da8f027c773fd547798a907fa94ca.jpg)



Fig. 6| Overview of Research Methodology. This figure depicts our three-stage approach, starting with data initialization, proceeding to the simulation phase, and culminating in model evaluation, where we validate our hypotheses.


# Reference



1. Yong, E. Opinion | Reporting on Long Covid Taught Me to Be a Better Journalist. The New York Times (2023). 





2. Smith, P. et al. Post COVID-19 condition and its physical, mental and social implications: protocol of a 2-year longitudinal cohort study in the Belgian adult population. Arch. Public Health Arch. Belg. Sante Publique 80, 151 (2022). 





3. Granovetter, M. S. The Strength of Weak Ties. Am. J. Sociol. 78, 1360–1380 (1973). 





4. McPherson, M., Smith-Lovin, L. & Cook, J. M. Birds of a Feather: Homophily in Social Networks. Annu. Rev. Sociol. 27, 415–444 (2001). 





5. Kossinets, G. & Watts, D. J. Origins of Homophily in an Evolving Social Network. Am. J. Sociol. 115, 405–450 (2009). 





6. Khanam, K. Z., Srivastava, G. & Mago, V. The homophily principle in social network analysis: A survey. Multimed. Tools Appl. 82, 8811–8854 (2023). 





7. Wettstein, M. Simulating hidden dynamics: Introducing Agent-Based Models as a tool for linkage analysis. Comput. Commun. Res. 2, 1–33 (2020). 





8. Fan, R., Xu, K. & Zhao, J. An agent-based model for emotion contagion and competition in online social media. Phys. Stat. Mech. Its Appl. 495, 245–259 (2018). 





9. Li, S., Liu, Z. & Li, Y. Temporal and spatial evolution of online public sentiment on emergencies. Inf. Process. Manag. 57, 102177 (2020). 





10. Hao, X., An, H., Zhang, L., Li, H. & Wei, G. Sentiment Diffusion of Public Opinions about Hot Events: Based on Complex Network. PLOS ONE 10, e0140027 





(2015). 





11. Waldherr, A. & Wettstein, M. Computational Communication Science| Bridging the Gaps: Using Agent-Based Modeling to Reconcile Data and Theory in Computational Communication Science. Int. J. Commun. 13, 24 (2019). 





12. Vartanian, K. et al. Integrating patient-reported physical, mental, and social impacts to classify long COVID experiences. Sci. Rep. 13, 16288 (2023). 





13. Byrne, E. A. Understanding Long Covid: Nosology, social attitudes and stigma. Brain. Behav. Immun. 99, 17–24 (2022). 





14. Brüssow, H. & Timmis, K. COVID-19: long covid and its societal consequences. Environ. Microbiol. 23, 4077–4091 (2021). 





15. Awoyemi, T., Ebili, U., Olusanya, A., Ogunniyi, K. E. & Adejumo, A. V. Twitter Sentiment Analysis of Long COVID Syndrome. Cureus 14, e25901. 





16. Damant, R. W. et al. Reliability and validity of the post COVID-19 condition stigma questionnaire: A prospective cohort study. EClinicalMedicine 55, 101755 (2023). 





17. Chen, X. et al. Negative Emotion Arousal and Altruism Promoting of Online Public Stigmatization on COVID-19 Pandemic. Front. Psychol. 12, (2021). 





18. Samper-Pardo, M. et al. The emotional well-being of Long COVID patients in relation to their symptoms, social support and stigmatization in social and health services: a qualitative study. BMC Psychiatry 23, 68 (2023). 





19. Sher, L. Post-COVID syndrome and suicide risk. QJM Mon. J. Assoc. Physicians 114, 95–98 (2021). 





20. Nabi, R. L. Emotional Flow in Persuasive Health Messages. Health Commun. 30, 114–124 (2015). 





21. Nabi, R. L. & Green, M. C. The Role of a Narrative’s Emotional Flow in Promoting Persuasive Outcomes. Media Psychol. 18, 137–162 (2015). 





22. Bailey, R. L., Wang, T. (Grace) & Kaiser, C. K. Clash of the Primary Motivations: Motivated Processing of Emotionally Experienced Content in Fear Appeals About Obesity Prevention. Health Commun. 33, 111–121 (2018). 





23. Berger, C. R. Uncertain Outcome Values in Predicted Relationships: Uncertainty Reduction Theory Then and Now. Hum. Commun. Res. 13, 34–38 (1986). 





24. Gudykunst, W. D. Anxiety/uncertainty management (AUM) theory: Current status. in Intercultural communication theory 8–58 (Sage Publications, Inc, Thousand Oaks, CA, US, 1995). 





25. Fernandez, G. et al. Social Network Analysis of COVID-19 Sentiments: 10 Metropolitan Cities in Italy. Int. J. Environ. Res. Public. Health 19, 7720 (2022). 





26. Crocamo, C. et al. Surveilling COVID-19 Emotional Contagion on Twitter by Sentiment Analysis. Eur. Psychiatry 64, e17 (2021). 





27. Iglesias-Sánchez, P. P., Vaccaro Witt, G. F., Cabrera, F. E. & Jambrino-Maldonado, C. The Contagion of Sentiments during the COVID-19 Pandemic Crisis: The Case of Isolation in Spain. Int. J. Environ. Res. Public. Health 17, 5918 (2020). 





28. Chen, X. & Yik, M. The Emotional Anatomy of the Wuhan Lockdown: Sentiment Analysis Using Weibo Data. JMIR Form. Res. 6, e37698 (2022). 





29. Li, S., Wang, Y., Xue, J., Zhao, N. & Zhu, T. The Impact of COVID-19 Epidemic Declaration on Psychological Consequences: A Study on Active Weibo Users. Int. J. Environ. Res. Public. Health 17, 2032 (2020). 





30. Matharaarachchi, S., Domaratzki, M., Katz, A. & Muthukumarana, S. Discovering 





Long COVID Symptom Patterns: Association Rule Mining and Sentiment Analysis in Social Media Tweets. JMIR Form. Res. 6, e37984 (2022). 





31. Yang, G., Wang, Z. & Chen, L. Investigating the Public Sentiment in Major Public Emergencies Through the Complex Networks Method: A Case Study of COVID-19 Epidemic. Front. Public Health 10, (2022). 





32. Kang, G. J. et al. Semantic network analysis of vaccine sentiment in online social media. Vaccine 35, 3621–3638 (2017). 





33. Rahman, M. M., Ali, G. G. M. N., Li, X. J., Paul, K. C. & Chong, P. H. J. Twitter and Census Data Analytics to Explore Socioeconomic Factors for Post-COVID-19 Reopening Sentiment. SSRN Scholarly Paper at https://doi.org/10.2139/ssrn.3639551 (2020). 





34. Pitroda, H. Long Covid Sentiment Analysis of Twitter Posts to understand public concerns. in 2022 8th International Conference on Advanced Computing and Communication Systems (ICACCS) vol. 1 140–148 (2022). 





35. Li, L., Hua, L. & Gao, F. What We Ask about When We Ask about Quarantine? Content and Sentiment Analysis on Online Help-Seeking Posts during COVID-19 on a Q&A Platform in China. Int. J. Environ. Res. Public. Health 20, 780 (2023). 





36. Esener, Y., McCall, T., Lakdawala, A. & Kim, H. Seeking and Providing Social Support on Twitter for Trauma and Distress During the COVID-19 Pandemic: Content and Sentiment Analysis. J. Med. Internet Res. 25, e46343 (2023). 





37. Ning, P. et al. COVID-19–Related Rumor Content, Transmission, and Clarification Strategies in China: Descriptive Study. J. Med. Internet Res. 23, e27339 (2021). 





38. Wang, D. & Qian, Y. Echo Chamber Effect in Rumor Rebuttal Discussions About COVID-19 in China: Social Media Content and Network Analysis Study. J. Med. Internet Res. 23, e27009 (2021). 





39. Dong, W. et al. Public Emotions and Rumors Spread During the COVID-19 Epidemic in China: Web-Based Correlation Study. J. Med. Internet Res. 22, e21933 (2020). 





40. Wang, P., Shi, H., Wu, X. & Jiao, L. Sentiment Analysis of Rumor Spread Amid COVID-19: Based on Weibo Text. Healthcare 9, 1275 (2021). 





41. Reiter-Haas, M., Klösch, B., Hadler, M. & Lex, E. Polarization of Opinions on COVID-19 Measures: Integrating Twitter and Survey Data. Soc. Sci. Comput. Rev. 41, 1811–1835 (2023). 





42. Jiang, J., Chen, E., Yan, S., Lerman, K. & Ferrara, E. Political polarization drives online conversations about COVID-19 in the United States. Hum. Behav. Emerg. Technol. 2, 200–211 (2020). 





43. Dye, T. D. et al. Risk of COVID-19-related bullying, harassment and stigma among healthcare workers: an analytical cross-sectional global study. BMJ Open 10, e046620 (2020). 





44. Liu, P. L. COVID-19 Information Seeking on Digital Media and Preventive Behaviors: The Mediation Role of Worry. Cyberpsychology Behav. Soc. Netw. 23, 677–682 (2020). 





45. Neely, S., Eldredge, C. & Sanders, R. Health Information Seeking Behaviors on Social Media During the COVID-19 Pandemic Among American Social 





Networking Site Users: Survey Study. J. Med. Internet Res. 23, e29802 (2021). 





46. An, L. et al. Measuring and profiling the topical influence and sentiment contagion of public event stakeholders. Int. J. Inf. Manag. J. Inf. Prof. 58, (2021). 





47. Ahmad, A. R. & Murad, H. R. The Impact of Social Media on Panic During the COVID-19 Pandemic in Iraqi Kurdistan: Online Questionnaire Study. J. Med. Internet Res. 22, e19556 (2020). 





48. Myrick, J. G. & Nabi, R. L. Fear Arousal and Health and Risk Messaging. in Oxford Research Encyclopedia of Communication (2017). doi:10.1093/acrefore/9780190228613.013.266. 





49. Leiss, W., Beck, U., Ritter, M., Lash, S. & Wynne, B. Risk Society, Towards a New Modernity. Can. J. Sociol. Cah. Can. Sociol. 19, 544 (1995). 





50. Li, Y., Hills, T. & Hertwig, R. A brief history of risk. Cognition 203, 104344 (2020). 





51. Carretié, L. Exogenous (automatic) attention to emotional stimuli: A review. Cogn. Affect. Behav. Neurosci. 14, 1228–1258 (2014). 





52. Heffner, J., Vives, M.-L. & FeldmanHall, O. Emotional responses to prosocial messages increase willingness to self-isolate during the COVID-19 pandemic. Personal. Individ. Differ. 170, 110420 (2021). 





53. Stijačić, M. P., Mišić, K. & Đurđević, D. F. Flattening the curve: COVID-19 induced a decrease in arousal for positive and an increase in arousal for negative words. Appl. Psycholinguist. 44, 1069–1089 (2023). 





54. Lekkas, D., Gyorda, J. A., Price, G. D., Wortzman, Z. & Jacobson, N. C. Using the COVID-19 Pandemic to Assess the Influence of News Affect on Online Mental Health-Related Search Behavior Across the United States: Integrated Sentiment 





Analysis and the Circumplex Model of Affect. J. Med. Internet Res. 24, e32731 (2022). 





55. Pangallo, M. et al. The unequal effects of the health–economy trade-off during the COVID-19 pandemic. Nat. Hum. Behav. 1–12 (2023) doi:10.1038/s41562-023- 01747-x. 





56. Changing emotions in the COVID-19 pandemic: A four-wave longitudinal study in the United States and China. Soc. Sci. Med. 285, 114222 (2021). 





57. Martinelli, N. et al. Time and Emotion During Lockdown and the Covid-19 Epidemic: Determinants of Our Experience of Time? Front. Psychol. 11, (2021). 





58. Shannon, C. E. A Mathematical Theory of Communication. Bell Syst. Tech. J. 27, 379–423 (1948). 





59. Guback, T. H. & DeFleur, M. L. Theories of Mass Communication. J. Aesthetic Educ. 2, 135 (1968). 





60. Bisgin, H., Agarwal, N. & Xu, X. A study of homophily on social media. World Wide Web 15, 213–232 (2012). 





61. Complex Spreading Phenomena in Social Systems. (Springer International Publishing, Cham, 2018). doi:10.1007/978-3-319-77332-2. 





62. Abreu, L. & Jeon, D.-S. Homophily in Social Media and News Polarization. SSRN Scholarly Paper at https://doi.org/10.2139/ssrn.3468416 (2019). 





63. Fersini, E., Pozzi, F. A. & Messina, E. Approval network: a novel approach for sentiment analysis in social networks. World Wide Web 20, 831–854 (2017). 





64. Boston University, Watts, S., Zhang, W., & University of Massachusetts Boston, USA. Capitalizing on Content: Information Adoption in Two Online communities. 





J. Assoc. Inf. Syst. 9, 73–94 (2008). 





65. Zhang, L., Li, H. & Chen, K. Effective Risk Communication for Public Health Emergency: Reflection on the COVID-19 (2019-nCoV) Outbreak in Wuhan, China. Healthcare 8, 64 (2020). 





66. Wu, Y., Xiao, H. & Yang, F. Government information disclosure and citizen coproduction during COVID-19 in China. Governance 35, 1005–1027 (2022). 





67. Lei, Y.-W. The Political Consequences of the Rise of the Internet: Political Beliefs and Practices of Chinese Netizens. Polit. Commun. 28, 291–322 (2011). 





68. Sullivan, J. China’s Weibo: Is faster different? New Media Soc. 16, 24–37 (2014). 





69. Newcomb, T. M. An approach to the study of communicative acts. Psychol. Rev. 60, 393–404 (1953). 





70. Festinger, L. A Theory of Cognitive Dissonance. xi, 291 (Stanford University Press, 1957). 





71. Windahl, D. M., Sven. Communication Models for the Study of Mass Communications. (Routledge, London, 2013). doi:10.4324/9781315846378. 





72. Petty, R., Wegener, D. & Fabrigar, L. Attitudes and Attitude Change. Annu. Rev. Psychol. 48, 609–47 (1997). 





73. McCOMBS, M. E. & SHAW, D. L. THE AGENDA-SETTING FUNCTION OF MASS MEDIA*. Public Opin. Q. 36, 176–187 (1972). 





74. Lasswell, H. The structure and function of communication in society. in (2007). 





75. Entman, R. M. Framing: Toward Clarification of a Fractured Paradigm. J. Commun. 43, 51–58 (1993). 





76. Scheufele, D. Framing as a theory of media effects. J. Commun. 49, 103–122 





(1999). 





77. Stockmann, D. Media Commercialization and Authoritarian Rule in China. (Cambridge University Press, Cambridge, 2012). doi:10.1017/CBO9781139087742. 





78. Barbieri, N., Bonchi, F. & Manco, G. Topic-Aware Social Influence Propagation Models. in 2012 IEEE 12th International Conference on Data Mining 81–90 (2012). doi:10.1109/ICDM.2012.122. 





79. Couldry, N. & Turow, J. Advertising, Big Data and the Clearance of the Public Realm: Marketers’ New Approaches to the Content Subsidy. Int. J. Commun. 8, 17 (2014). 





80. Molyneux, L. What journalists retweet: Opinion, humor, and brand development on Twitter. Journalism 16, 920–935 (2015). 





81. Allcott, H. & Gentzkow, M. Social Media and Fake News in the 2016 Election. J. Econ. Perspect. 31, 211–236 (2017). 





82. van Dijck, J. & Poell, T. Understanding Social Media Logic. SSRN Scholarly Paper at https://papers.ssrn.com/abstrac $\underline { { \underline { { \mathbf { \Pi } } } } }$ 2309065 (2013). 





83. Fan, R., Zhao, J., Chen, Y. & Xu, K. Anger Is More Influential than Joy: Sentiment Correlation in Weibo. PLOS ONE 9, e110184 (2014). 





84. Wollebæk, D., Karlsen, R., Steen-Johnsen, K. & Enjolras, B. Anger, Fear, and Echo Chambers: The Emotional Basis for Online Behavior. Soc. Media Soc. 5, 2056305119829859 (2019). 





85. Fan, R., Xu, K. & Zhao, J. Higher contagion and weaker ties mean anger spreads faster than joy in social media. Preprint at 





https://doi.org/10.48550/arXiv.1608.03656 (2016). 





86. Jiang, Y. ‘Reversed agenda-setting effects’ in China Case studies of Weibo trending topics and the effects on state-owned media in China. J. Int. Commun. 20, 168–183 (2014). 





87. McGinty, E. E., Presskreischer, R., Han, H. & Barry, C. L. Psychological Distress and Loneliness Reported by US Adults in 2018 and April 2020. JAMA 324, 93 (2020). 





88. Zhao, L. et al. Sentiment contagion in complex networks. Phys. Stat. Mech. Its Appl. 394, 17–23 (2014). 





89. Kozitsin, I. V. A general framework to link theory and empirics in opinion formation models. Sci. Rep. 12, 5543 (2022). 





90. Lewis, K., Gonzalez, M. & Kaufman, J. Social selection and peer influence in an online social network. PNAS Proc. Natl. Acad. Sci. U. S. Am. 109, 68–72 (2012). 





91. Measuring and profiling the topical influence and sentiment contagion of public event stakeholders. Int. J. Inf. Manag. 58, 102327 (2021). 





92. Casini, L. & Manzo, G. Agent-Based Models and Causality : A Methodological Appraisal. (Linköping University Electronic Press, 2016). 





93. Hedström, P. & Ylikoski, P. Causal Mechanisms in the Social Sciences. Annu. Rev. Sociol. 36, 49–67 (2010). 





94. Zhou, T., Nguyen, T.-V. T., Zhong, J. & Liu, J. A COVID-19 descriptive study of life after lockdown in Wuhan, China. R. Soc. Open Sci. 7, 200705 (2020). 





95. Yu, S., Eisenman, D. & Han, Z. Temporal Dynamics of Public Emotions During the COVID-19 Pandemic at the Epicenter of the Outbreak: Sentiment Analysis of 





Weibo Posts From Wuhan. J. Med. Internet Res. 23, e27078 (2021). 





96. Király, O. et al. Preventing problematic internet use during the COVID-19 pandemic: Consensus guidance. Compr. Psychiatry 100, 152180 (2020). 





97. Wilson, O. & Flahault, A. China’s U-turn in its COVID-19 policy. Anaesth. Crit. Care Pain Med. 42, 101197 (2023). 





98. Pengpeng, L., Fangqi, Z. & Qianru, Z. Communication Mechanisms and Implications of the COVID-19 Risk Event in Chinese Online Communities. Front. Public Health 10, (2022). 





99. Karatzas, E., Baltoumas, F. A., Panayiotou, N. A., Schneider, R. & Pavlopoulos, G. A. Arena3Dweb: interactive 3D visualization of multilayered networks. Nucleic Acids Res. 49, W36–W45 (2021). 





100. Zhang, W., Chen, Z. & Xi, Y. Traffic media: how algorithmic imaginations and practices change content production. Chin. J. Commun. 14, 58–74 (2020). 





101. Ludwig, K. et al. Divided by the Algorithm? The (Limited) Effects of Contentand Sentiment-Based News Recommendation on Affective, Ideological, and Perceived Polarization. Soc. Sci. Comput. Rev. 41, 2188–2210 (2023). 





102. Stieglitz, S. & Dang-Xuan, L. Emotions and Information Diffusion in Social Media—Sentiment of Microblogs and Sharing Behavior. J. Manag. Inf. Syst. 29, 217–248 (2013). 





103. Hsu, P.-Y. et al. Effects of sentiment on recommendations in social network. Electron. Mark. 29, 253–262 (2019). 





104. Lang, A. The Limited Capacity Model of Mediated Message Processing. J. Commun. 50, 46–70 (2000). 





105. Dubey, S. et al. Psychosocial impact of COVID-19. Diabetes Metab. Syndr. 14, 779–788 (2020). 





106. Xu, L., Lin, H., Pan, Y. & al, et. Construction of Emotion Lexicon Ontology. J. Intell. 27, 180–185 (2008). 





107. Müller, P., Chan, C.-H., Ludwig, K., Freudenthaler, R. & Wessler, H. Differential Racism in the News: Using Semi-Supervised Machine Learning to Distinguish Explicit and Implicit Stigmatization of Ethnic and Religious Groups in Journalistic Discourse. Polit. Commun. 40, 396–414 (2023). 

