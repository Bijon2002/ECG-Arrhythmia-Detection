# FINALIZED VIVA VOCE PRESENTATION SCRIPT
**Student:** Bijosilin Marisilin | **Student ID:** 2541518  
**Project:** Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification  
**Supervisor:** Ms. Niroji Thayalan | **Unit Leader:** Enjie Liu  
**Degree:** BSc (Hons) Computer Science | **Unit:** CIS017-3 Undergraduate Project  
**Institutions:** University of Bedfordshire (UK) & Northern Uni (SLIIT)  
**Total Target Time:** 8 to 10 Minutes  

---

### [00:00 - 01:15] SECTION 1: INTRODUCTION & THE CORE INSPIRATION

Good morning, respected examiners and members of the evaluation panel.

Hi, I'm **Bijosilin Marisilin**, student ID **2541518**. 

Today, I am proud to present my undergraduate final project titled: **"Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification"**, carried out under the supervision of **Ms. Niroji Thayalan** for unit **CIS017-3** at the **University of Bedfordshire** in academic partnership with **Northern Uni**.

Let me start by sharing exactly how I got the idea for this research.

During my study of applied machine learning in healthcare, I noticed a very disturbing problem. In published research papers, deep neural networks often report 98% or even 99% accuracy on cardiac benchmarks. Yet, when you speak with cardiologists in hospital intensive care units, they refuse to let these models make autonomous clinical decisions.

When I investigated why, I discovered a critical medical vulnerability known as **"Softmax Overconfidence"**. 

Standard neural networks are mathematically forced to output a classification, no matter how corrupted, noisy, or unfamiliar the incoming heartbeat is. If an ambulatory patient rolls over in bed, causing an electrode lead to slip and introduce high-amplitude muscular noise, a standard deep learning model will not say *"I am unsure"*. Instead, it will analyze the noisy artifact and output a diagnosis like *"Ventricular Tachycardia"* with 99% false confidence.

Furthermore, in real clinical datasets, lethal arrhythmias like Fusion beats are rare, making up less than 1% of recordings, while healthy heartbeats are abundant. Standard models become biased toward the normal majority, silently missing the rare life-threatening beats.

This inspired my central research mission:  
**Can we engineer an AI system that not only classifies heartbeats with high precision, but is also self-aware enough to quantify its own uncertainty, separate sensor noise from clinical ambiguity, and safely trigger an automated refusal protocol when patient safety is at risk?**

---

### [01:15 - 02:45] SECTION 2: DATASET SOURCE & SIGNAL PREPROCESSING

To solve this problem with real clinical validity, I turned to the gold-standard benchmark in cardiac electrophysiology: the **MIT-BIH Arrhythmia Database**, which I acquired from **PhysioNet**, a research repository established by MIT and Beth Israel Hospital in Boston, supported by the National Institutes of Health.

This database contains **48 continuous, half-hour, two-channel ambulatory Holter ECG recordings** collected from 47 human subjects. The analog signals were digitized at a clinical sampling rate of **360 samples per second (360 Hz)** with 11-bit resolution over a 10 mV range. Most importantly, every single heartbeat in this dataset was independently annotated and verified by at least two certified cardiologists. For my study, I extracted **Lead MLII (Modified Limb Lead II)**, as it aligns directly with the heart’s electrical axis and provides the clearest view of the P-wave and QRS complex.

Raw ECG signals directly from a patient are contaminated with noise. To engineer clinical-grade training data, I built a 5-step preprocessing pipeline in Python using the `wfdb` library:

* **Step 1 was Noise Elimination:** Raw signals suffer from low-frequency baseline drift caused by patient respiration and perspiration, as well as high-frequency electromyographic muscle tremor. I applied a zero-phase **2nd-order Butterworth Bandpass Filter with cutoff frequencies of 0.5 Hz and 45 Hz**, removing baseline wander and mains hum while keeping the critical QRS morphology pristine.
* **Step 2 was R-Peak Centering and Segmentation:** Around each verified R-peak annotation, I extracted a **360-sample window**—corresponding to exactly 1.0 second of cardiac duration, with 90 samples before the peak and 270 samples after, capturing the full P-QRS-T cycle.
* **Step 3 was Z-Score Standardization:** To eliminate inter-patient voltage differences caused by varied skin impedance, each heartbeat was normalized to zero mean and unit variance.
* **Step 4 was International Standard Mapping:** Following the **ANSI/AAMI EC57 standard**, all 109,446 heartbeats were mapped into 5 universal clinical categories:
  - Class N: Normal beats (90,589 beats)
  - Class S: Supraventricular ectopic beats (2,779 beats)
  - Class V: Ventricular ectopic beats (7,236 beats)
  - Class F: Fusion of ventricular and normal beats (803 beats — our critical minority)
  - Class Q: Unknown or paced beats (8,039 beats)
* **Step 5 was a Stratified Train-Test Split:** 80% (87,558 beats) were allocated for training, and 20% (21,888 beats) were held out strictly for independent final testing.

---

### [02:45 - 04:30] SECTION 3: 1D-CNN ARCHITECTURE & TRUE DEEP ENSEMBLE TRAINING

Now, let me walk you through the model architecture and how I trained the system.

Many literature approaches convert 1D ECG waveforms into 2D spectrogram images and pass them into heavy computer vision models like ResNet-50. I avoided this design because converting 1D electrical voltages into 2D pictures introduces lossy interpolation, destroys exact temporal alignment, and creates massive computational overhead.

Instead, I designed an optimized, native **1D-Convolutional Neural Network (1D-CNN)**:
* **Input Layer:** Directly accepts a temporal vector of shape `(Batch_Size, 1, 360)`.
* **Convolutional Block 1:** Consists of 32 filters with a kernel size of 5, followed by 1D Batch Normalization, ReLU activation, Max Pooling of stride 2 which reduces the temporal dimension to 180, and Spatial Dropout of 0.2.
* **Convolutional Block 2:** Consists of 64 filters of kernel size 5, with Batch Normalization, ReLU, Max Pooling reducing temporal dimension to 90, and 0.2 Dropout.
* **Convolutional Block 3:** Consists of 128 filters of kernel size 3 to extract fine morphological wave contours, followed by Batch Normalization, ReLU, Max Pooling reducing temporal dimension to 45, and 0.3 Dropout.
* **Classification Head:** The 45 by 128 feature tensor is flattened into a 5,760-dimensional vector, fed through a dense linear layer of 128 hidden neurons with 0.4 Dropout, and terminates in a final linear projection outputting 5 unnormalized class logits.

To eliminate single-model bias, I trained a **True Deep Ensemble of 5 distinct 1D-CNN models**.

Here is how I conducted the training step by step:
1. **Diversity through Random Initialization:** Rather than training one network, I initialized 5 separate models using **5 distinct random seeds: 42, 101, 202, 303, and 404**. This forced each model's weights to start in completely different regions of the loss landscape and encounter mini-batches in different permutations.
2. **Handling Severe Class Imbalance:** To prevent the 90,000 Normal beats from drowning out the 803 Fusion beats, I applied **Class-Weighted Cross-Entropy Loss**, weighting each class inversely proportional to its training frequency. Misclassifying a rare Fusion beat incurred over 100 times greater gradient penalty than misclassifying a Normal beat.
3. **Training Dynamics:** I trained all 5 models using the Adam optimizer with a learning rate of 0.001 and a batch size of 64 over 5 epochs. As shown in **Figure 4.1**, all 5 models converged with remarkable consistency, dropping from an initial loss of approximately 0.39 to 0.41 down to a final converged loss between 0.127 and 0.142, with a mean ensemble loss of 0.1327 and zero overfitting.

Each model was saved as an independent checkpoint for ensemble inference.

---

### [04:30 - 06:15] SECTION 4: 3-TIER UNCERTAINTY QUANTIFICATION & CALIBRATION

Once the ensemble was trained, I engineered the core scientific contribution of this thesis: **The 3-Tier Uncertainty Quantification Pipeline**.

When an ECG beat arrives, it passes through three complementary uncertainty checks:

* **Tier 1 — Epistemic Model Uncertainty via Monte Carlo Dropout:**  
  By keeping dropout active at inference time and executing $T = 15$ stochastic forward passes per ensemble member, we calculate the variance across the predictions. If the model has never encountered this specific waveform shape during training, the stochastic subnetworks disagree, producing high variance.
* **Tier 2 — Aleatoric Data Uncertainty via Predictive Entropy:**  
  We calculate the Shannon Entropy across the ensemble's mean softmax distribution:  
  $$H(p) = -\sum_{c=1}^5 p_c \log_2(p_c)$$  
  When incoming electrical signals have overlapping class boundaries or baseline noise, the probability distribution spreads across multiple classes, causing predictive entropy to spike.
* **Tier 3 — My Novel Contribution: Cluster-Based Entropy (CBE):**  
  Standard Shannon entropy has a major medical flaw: it treats all classification errors equally. In an intensive care unit, if an algorithm is slightly unsure whether a beat is Normal or a harmless Supraventricular ectopic, raising an alarm creates alarm fatigue. However, if the algorithm is unsure between a Normal beat and a lethal Ventricular ectopic, missing it could allow cardiac arrest.  
  To solve this, I formulated **Cluster-Based Entropy (CBE)** by grouping the 5 AAMI classes into **4 Clinical Risk Tiers**:
  1. Tier 1: Normal $\{N\}$
  2. Tier 2: Supraventricular $\{S\}$
  3. Tier 3: Malignant Ventricular and Fusion $\{V, F\}$
  4. Tier 4: Unknown / Paced $\{Q\}$  
  CBE computes entropy strictly across these risk boundaries. Ambiguity within the same clinical risk cluster produces zero cluster entropy, whereas any ambiguity that crosses into the ventricular cluster triggers an immediate high-priority alert.
* **Post-Hoc Temperature Calibration:**  
  Even an ensemble can produce overconfident raw probabilities. I optimized a post-hoc Temperature parameter $T = 1.48$ on validation logits. Because dividing logits by a positive scalar is strictly monotonic, it preserves 100% of our classification accuracy while softening overconfidence, dropping our **Expected Calibration Error (ECE) from 0.084 down to 0.018**.

---

### [06:15 - 08:00] SECTION 5: EMPIRICAL RESULTS & LIVE CLINICAL DASHBOARD DEMO

We evaluated our complete system on the **21,888 independent test beats**, producing clear empirical improvements over baseline single-model implementations:

* **Overall Test Accuracy:** Reached **94.0%**, compared to 90.4% for a baseline single 1D-CNN.
* **Macro Average F1-Score:** Rose from **0.74 to 0.83**.
* **Rare Fusion Beat F1 (Class F):** Crucially jumped from **0.40 to 0.59**—an absolute improvement of **+19.0%**! This proves that our class-weighted loss and deep ensemble successfully rescued the rarest and most dangerous arrhythmias from being missed.
* **Ventricular Ectopic Recall (Class V):** Achieved **98% Sensitivity**, ensuring dangerous premature ventricular contractions are reliably detected.

**Clinical Decision Support System (The Dashboard):**  
To demonstrate how this translates into hospital practice, I developed a production-ready, full-stack clinical interface using **Flask, HTML5 Canvas, and Web Audio API**:

*(Here, glance at or point toward your demo screen)*

1. **Live 360 Hz Telemetry Oscilloscope:** The monitor renders continuous, streaming ECG waveforms with animated P-Q-R-S-T peaks and a dynamic heart rate counter.
2. **Interactive Heartbeat Testing:** A clinician can click to inject Normal, Ventricular, Fusion, or Out-Of-Distribution (OOD) noise beats in real time.
3. **3D Uncertainty HUD:** Displays the three uncertainty gauges side-by-side with color-coded safety thresholds.
4. **Web Audio Sonification:** The system generates realistic pulse beeps matching the patient's heart rate, automatically shifting pitch when a ventricular arrhythmia occurs to give instant auditory feedback.
5. **Automated Safety Fallback Protocol:** If an electrode falls off or high epistemic variance is detected, the AI refuses to guess, halts autonomous prediction, and displays a red emergency alert:  
   *"CRITICAL CLINICAL SAFETY PROTOCOL TRIGGERED: Automated diagnosis suspended. Human Cardiologist Review Required."*
6. **50-Beat Holter Batch Engine:** Cardiologists can upload a multi-beat patient recording and immediately receive a color-coded triage table sorting beats by diagnostic risk and uncertainty.

---

### [08:00 - 08:45] SECTION 6: CONCLUSION & FUTURE WORK

To conclude, this project demonstrates that we do not have to accept opaque, overconfident black-box AI in healthcare. 

By combining an optimized **5-member True Deep Ensemble**, our novel **3-tier Uncertainty Framework with Cluster-Based Entropy**, and post-hoc **Temperature Calibration**, we have created an ECG diagnostic system that is accurate, well-calibrated, and humble enough to know its own limits.

For future work, I plan to:
1. Port this trained PyTorch ensemble to low-cost **STM32 and ESP32 microcontrollers using ONNX Runtime** to enable ultra-low-power edge deployment on wearable Holter patches.
2. Extend the architecture from single-lead MLII to full **12-lead multi-channel ECGs using Spatial-Temporal Transformers** for acute myocardial infarction (STEMI) detection.

Thank you very much for your time and kind consideration. I am now delighted to answer any questions from the panel.
