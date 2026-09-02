# Comprehensive Viva Presentation Script & Defense Guide
**Project Title:** Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification  
**Student:** Bijosilin Marisilin (Student ID: 2541518)  
**Supervisor:** Ms. Niroji Thayalan  
**Unit:** CIS017-3 Undergraduate Project  
**Institution:** University of Bedfordshire (UK) in partnership with Northern Uni (SLIIT)

---

## ⏱️ Presentation Timing Breakdown (Approx. 8–10 Minutes)
1. **Introduction & Motivation:** 1.5 mins
2. **Research Problem & Objectives:** 1.5 mins
3. **Dataset & Preprocessing:** 1.5 mins
4. **Proposed Methodology & 3-Tier UQ:** 2.5 mins
5. **Key Results & Clinical Dashboard Demo:** 2 mins
6. **Conclusion & Future Directions:** 1 min
7. **Viva Q&A Defense Session:** (Panel Questions)

---

## 🎙️ Full Spoken Script (Word-for-Word in Simple, Clear English)

### 1. Opening & Introduction
> "Good morning / afternoon respected examiners and members of the evaluation panel.
>
> Hi, I'm **Bijosilin Marisilin**, student ID **2541518**. 
> Today, I am proud to present my undergraduate final project titled: **'Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification'**, developed under the supervision of **Ms. Niroji Thayalan** for the module **CIS017-3** at the **University of Bedfordshire** in partnership with **Northern Uni**.
>
> Let me start with a simple question: *Would a doctor trust an AI that claims 99% confidence on a completely corrupted or rare heart signal?*
> In clinical practice, the answer is an absolute **no**."

---

### 2. Clinical Problem & The "AI Safety Gap"
> "Cardiovascular diseases are the world’s leading cause of death, taking over **17.9 million lives** each year. Continuous 24-hour ambulatory Holter monitoring is essential to catch silent arrhythmias before they become fatal.
>
> However, cardiologists cannot manually inspect every heartbeat across 100,000 beats per patient per day. Automated machine learning models are supposed to assist them. But standard neural networks have a dangerous flaw called **Softmax Overconfidence**. 
>
> Even when an ECG signal is corrupted by electrode movement, baseline wander, or a rare borderline heartbeat, traditional deep learning models force a single hard diagnosis with dangerously high confidence. They cannot tell the difference between:
> 1. What the model genuinely knows, and
> 2. What is simply noise or unfamiliar data.
>
> In healthcare, an overconfident wrong diagnosis can be fatal. This is the **AI Safety Gap** that my project directly solves."

---

### 3. Project Objectives
> "To address this critical gap, I set **five core engineering and research objectives**:
>
> * **First:** Preprocess and segment raw ECG telemetry into isolated single heartbeats following standard clinical frequencies.
> * **Second:** Design and train a diverse **5-member True Deep Ensemble** using class-weighted cross-entropy to handle extreme class imbalance.
> * **Third:** Engineer a novel **3-Tier Uncertainty Quantification (UQ)** framework incorporating Monte Carlo Dropout, Shannon Predictive Entropy, and a new clinical metric I formulated called **Cluster-Based Entropy (CBE)**.
> * **Fourth:** Calibrate the ensemble’s probability outputs using post-hoc **Temperature Scaling** to eliminate overconfidence and minimize the Expected Calibration Error (ECE).
> * **Fifth:** Build and deploy a real-time, interactive **Clinical Decision-Support Web Dashboard** equipped with live 360 Hz telemetry, audio sonification beeps, and an automatic safety refusal trigger."

---

### 4. Dataset & Preprocessing Pipeline
> "For our benchmark data, I utilized the gold-standard **PhysioNet MIT-BIH Arrhythmia Database**, consisting of 48 half-hour ambulatory Holter recordings sampled at **360 Hz** from Lead MLII. 
>
> The raw signals contain significant noise: baseline wander from patient breathing and high-frequency muscular tremor (EMG).
>
> To clean this:
> 1. I passed the signals through a zero-phase **2nd-order Butterworth Bandpass Filter (0.5 to 45 Hz)**.
> 2. Using annotated R-peak locations, I extracted **360-sample windows**—corresponding to 1 second of cardiac activity centered around each R-peak.
> 3. Each beat was normalized using **Z-score standardization** to remove patient-specific amplitude differences.
>
> The dataset comprises **109,446 heartbeats**, mapped to the official **AAMI EC57 5-Class standard**:
> - **N (Normal):** over 90,000 beats
> - **S (Supraventricular ectopic):** 2,779 beats
> - **V (Ventricular ectopic):** 7,236 beats
> - **F (Fusion of ventricular and normal):** 803 beats *(severe minority class)*
> - **Q (Unknown / Paced):** 8,039 beats
>
> I partitioned this into **87,558 training beats (80%)** and **21,888 test beats (20%)** for rigorous evaluation."

---

### 5. Model Architecture & True Deep Ensemble
> "Instead of relying on a single neural network, I designed a **True Deep Ensemble** consisting of **5 distinct 1D-Convolutional Neural Networks (1D-CNNs)**.
>
> Each individual network has 3 convolutional feature extraction blocks with batch normalization, ReLU activation, max pooling, and spatial dropout, followed by fully connected dense layers.
>
> What makes this a **True Ensemble**?
> * Each of the 5 models was initialized with a **different random seed** (Seeds 42, 101, 202, 303, and 404) and exposed to randomized batch orders.
> * This forces the models to explore completely different local minima on the non-convex loss surface.
> * When predicting, we aggregate the predictions across all 5 models. If all 5 agree, confidence is genuine. If they disagree, the disagreement itself signals clinical uncertainty."

---

### 6. Our Novelty: 3-Tier Uncertainty Quantification & CBE
> "The heart of my research contribution is the **3-Tier Uncertainty Framework**:
>
> 1. **Tier 1 - Epistemic Model Uncertainty (MC Dropout):**  
>    By keeping dropout active at inference time and running $T=15$ stochastic forward passes, we measure the variance across predictions. High variance means the model lacks knowledge about this specific waveform.
>
> 2. **Tier 2 - Aleatoric Data Noise (Predictive Entropy):**  
>    We calculate the Shannon Entropy on the softmax distribution. If the probabilities are split 50/50, predictive entropy spikes, detecting ambiguous beat morphology.
>
> 3. **Tier 3 - My Novel Contribution: Cluster-Based Entropy (CBE):**  
>    Standard Shannon entropy treats all classification confusion equally. But medically, confusing a Normal beat with a Supraventricular beat is minor, whereas confusing a Normal beat with a lethal Ventricular ectopic is an emergency!  
>    I grouped the 5 AAMI classes into **4 risk clusters**:
>    - Cluster 1: Normal $\{N\}$
>    - Cluster 2: Supraventricular $\{S\}$
>    - Cluster 3: Malignant Ventricular & Fusion $\{V, F\}$
>    - Cluster 4: Unknown / Paced $\{Q\}$
>    
>    CBE calculates entropy across these risk clusters. It prevents false alarms when benign classes overlap, but triggers an immediate clinical emergency alert whenever ventricular ambiguity occurs."

---

### 7. Temperature Calibration
> "Even an ensemble can produce miscalibrated probabilities. To fix this, I applied **Temperature Scaling** on the validation logits using a learned parameter $T = 1.48$. 
>
> It scales the logits before softmax without changing the classification ranking. This reduced our **Expected Calibration Error (ECE)** from **0.084 down to 0.018**. 
> This means when our system says it is 95% confident, it is statistically accurate 95% of the time."

---

### 8. Empirical Results & Performance Gains
> "Looking at our quantitative benchmarks evaluated on the **21,888 independent test beats**:
>
> * **Overall Accuracy:** Reached **94.0%** (up from 90.4% baseline).
> * **Macro Average F1-Score:** Increased from **0.74 to 0.83**.
> * **Rare Fusion Beat F1 (Class F):** Crucially jumped from **0.40 to 0.59**—an absolute improvement of **+19.0%**! This proves that our class-weighted loss and deep ensemble successfully rescued rare, lethal arrhythmias from being missed.
> * **Ventricular Sensitivity (Class V):** Achieved **98% Recall**, ensuring dangerous PVCs are not overlooked."

---

### 9. Clinical Web Dashboard & Practical Demonstration
> "To prove that this research works in the real world, I built an interactive, production-ready **Clinical Decision Support System**:
>
> * **Real-Time 360 Hz Oscilloscope:** Streams live continuous ECG waveforms with a dynamic BPM counter and P-Q-R-S-T peak markers.
> * **Interactive Waveform Injection:** A doctor can click to inject Normal, Ventricular, Fusion, or Out-Of-Distribution (OOD) noise beats in real time.
> * **3D Uncertainty Gauges & Radar:** Displays the 3 uncertainty gauges side-by-side with color-coded safety thresholds.
> * **Web Audio Sonification:** The dashboard synthesizes audio beeps matching the patient's heart rate, automatically shifting pitch when an abnormal ventricular beat is detected.
> * **Automated Safety Fallback:** If high epistemic uncertainty or electrode artifact is detected, the AI refuses to guess, halts autonomous diagnosis, and displays: *'CRITICAL CLINICAL SAFETY PROTOCOL TRIGGERED: Automated diagnosis suspended. Human Cardiologist Review Required.'*
> * **50-Beat Holter Batch Engine:** Cardiologists can upload a recording and instantly receive a color-coded triage table sorting beats by diagnostic risk and uncertainty."

---

### 10. Conclusion & Future Work
> "In conclusion, this project demonstrates that we do not have to settle for black-box neural networks in healthcare. By combining a **5-member True Deep Ensemble**, **3-tier Uncertainty Quantification**, and **Temperature Calibration**, we have created a transparent, safe, and clinically trustworthy ECG diagnostic system.
>
> For future work, I plan to:
> 1. Port the trained ensemble to low-power edge microcontrollers like STM32 or ESP32 using ONNX Runtime for wearable Holter patches.
> 2. Extend the architecture from single-lead MLII to full 12-lead ECGs using Spatial-Temporal Transformers for acute myocardial infarction (STEMI) detection.
>
> Thank you very much for your time and kind attention. I am now open to your questions."

---

## 🎯 Top 10 Tough Viva Questions & Winning Answers

### Q1: Why did you choose a 5-member True Deep Ensemble over Bayesian Neural Networks (BNNs) or MC Dropout alone?
**Answer:**
> "While BNNs provide theoretically sound posteriors, they double the parameter count, are notoriously difficult to converge via variational inference, and often suffer from posterior collapse. MC Dropout is fast, but because it only uses a single trained weight checkpoint, it explores only a local neighborhood around one minimum. True Deep Ensembles train distinct models from different random initializations. DeepMind and Lakshminarayanan et al. empirically proved that Deep Ensembles capture diverse modes in the loss landscape and are superior to BNNs in both calibration and out-of-distribution detection."

---

### Q2: Explain the difference between Aleatoric and Epistemic uncertainty in your ECG context.
**Answer:**
> "*Aleatoric uncertainty* is noise inherent in the data—such as electrode motion artifacts, baseline wander, or overlapping class boundaries between similar beats. It cannot be reduced by collecting more training data of the same type.  
> *Epistemic uncertainty* is model ignorance—when the AI encounters a rare arrhythmia or a pattern it was never trained on. This uncertainty can be reduced by feeding the model more relevant training data. Our system separates them using Predictive Entropy for aleatoric noise and MC Dropout variance across models for epistemic uncertainty."

---

### Q3: What is the clinical motivation behind your novel Cluster-Based Entropy (CBE)?
**Answer:**
> "Standard Shannon entropy treats all misclassifications with equal penalty. In a medical setting, if the model is slightly unsure whether a beat is a Normal beat ($N$) or a benign Supraventricular ectopic ($S$), raising a high-priority alarm leads to alarm fatigue. However, if the model is unsure between Normal ($N$) and a malignant Ventricular ectopic ($V$), missing it could allow ventricular fibrillation and cardiac arrest. CBE collapses the 5 classes into 4 clinical risk tiers ($\{N\}$, $\{S\}$, $\{V,F\}$, $\{Q\}$). Ambiguity inside the same cluster produces zero cluster entropy, whereas ambiguity crossing into the ventricular cluster triggers an immediate alarm."

---

### Q4: How did you select the hyperparameters for the Butterworth filter (0.5 to 45 Hz)?
**Answer:**
> "According to clinical cardiology guidelines and the American Heart Association (AHA), meaningful diagnostic ECG frequency content lies between 0.67 Hz and 40 Hz. The 0.5 Hz high-pass cutoff eliminates low-frequency baseline drift caused by patient respiration and perspiration without distorting the ST-segment. The 45 Hz low-pass cutoff suppresses 50/60 Hz electrical mains hum and high-frequency electromyographic (EMG) muscle tremor while preserving the rapid QRS complex slope."

---

### Q5: How does Temperature Scaling work and why does it improve calibration without hurting accuracy?
**Answer:**
> "Temperature Scaling introduces a single positive scalar $T$ to divide the output logits before applying the softmax function: $\hat{p}_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$.  
> Because $T$ is applied uniformly across all logits, the relative order (the argmax) remains identical, meaning classification accuracy is 100% preserved. When $T > 1$, it softens overconfident probabilities towards a uniform distribution. We optimized $T$ using Negative Log-Likelihood on validation data, achieving $T = 1.48$ and dropping ECE from 0.084 down to 0.018."

---

### Q6: Your dataset has 90,000 Normal beats and only 803 Fusion beats. How did you handle this extreme imbalance?
**Answer:**
> "Severe imbalance was tackled using two coordinated strategies:  
> First, **Class-Weighted Cross-Entropy Loss**, where the loss weight for class $c$ is inversely proportional to its frequency: $w_c = \frac{N}{K \cdot N_c}$. This penalized the model heavily whenever it misclassified rare Fusion or Supraventricular beats.  
> Second, the **Deep Ensemble consensus voting** combined with ensemble variance prevented the majority Normal class from washing out the minority predictions. As a result, our Fusion F1 score rose by $+19.0\%$ (from 0.40 to 0.59)."

---

### Q7: Why did you choose 360 samples per heartbeat window?
**Answer:**
> "The MIT-BIH database is sampled at 360 Hz. A window of 360 samples corresponds to exactly 1.0 second in real time. Because a normal human heart rate ranges between 60 to 100 beats per minute, 1.0 second centered on the R-peak (90 samples before the R-peak and 270 samples after) reliably captures the preceding P-wave, the entire QRS complex, and the following T-wave across diverse heart rates."

---

### Q8: What happens in your system when a patient disconnects an electrode lead?
**Answer:**
> "When an electrode lead falls off or suffers severe motion artifact, the resulting flatline or extreme high-amplitude noise is completely Out-Of-Distribution (OOD). The 5 ensemble models produce conflicting, high-variance outputs, and predictive entropy spikes above our safety threshold of 0.85. The safety gate immediately activates, pauses automated diagnosis, rings an audio alert, and instructs the medical staff to inspect the lead connection."

---

### Q9: Could this model run on an Apple Watch or an embedded wearable device?
**Answer:**
> "Yes. While training 5 deep models requires a GPU, inference on 1D-CNNs is lightweight. A 360-sample 1D vector requires negligible floating-point operations (FLOPs) compared to 2D image models. By converting our PyTorch ensemble to ONNX format and applying FP16 or INT8 post-training quantization, all 5 models can run in under 45 milliseconds on microcontrollers like an ARM Cortex-M4 (STM32) or ESP32."

---

### Q10: What is your primary takeaway from completing this undergraduate project?
**Answer:**
> "My key takeaway is that in safety-critical medical engineering, predictive accuracy alone is insufficient. An AI system must possess self-awareness of its own uncertainty. Building a system that knows *when it does not know* and safely defers to human experts is the true bridge between theoretical deep learning and real-world clinical adoption."
