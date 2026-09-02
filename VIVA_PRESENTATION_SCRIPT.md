# Complete Viva Presentation Script & Defense Guide (Full Step-by-Step Flow)
**Project Title:** Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification  
**Student:** Bijosilin Marisilin (Student ID: 2541518)  
**Supervisor:** Ms. Niroji Thayalan | **Unit Leader:** Enjie Liu  
**Unit:** CIS017-3 Undergraduate Project  
**Institution:** University of Bedfordshire (UK) in partnership with Northern Uni (SLIIT)

---

## ⏱️ Presentation Timing Breakdown (Approx. 10–12 Minutes)
1. **Introduction & How I Got This Idea:** 2.0 mins
2. **Where I Got The Dataset (PhysioNet MIT-BIH):** 1.5 mins
3. **Step-by-Step Data Engineering & Preprocessing:** 2.0 mins
4. **1D-CNN Model Architecture & Deep Ensemble:** 2.0 mins
5. **How I Trained the Models (Step-by-Step):** 2.0 mins
6. **Novel 3-Tier Uncertainty Pipeline & Temperature Calibration:** 1.5 mins
7. **Empirical Results & Live Dashboard Demo:** 1.5 mins
8. **Conclusion & Future Work:** 0.5 min
9. **Viva Panel Q&A Defense Session**

---

## 🎙️ Spoken Presentation Script (Word-for-Word Narrative Flow)

### 1. Opening & How I Got This Idea
> "Good morning / afternoon respected examiners and members of the evaluation panel.
>
> Hi, I'm **Bijosilin Marisilin**, student ID **2541518**. Today, I am excited to present my undergraduate final project: **'Uncertainty-Aware True Deep Ensemble System for Reliable ECG Arrhythmia Classification'**, conducted under the supervision of **Ms. Niroji Thayalan** for unit **CIS017-3** at the **University of Bedfordshire** and **Northern Uni**.
>
> **Let me share how I got this idea:**  
> During my research into medical artificial intelligence, I noticed a striking paradox. We have deep learning models boasting 98% or 99% accuracy on paper, yet doctors in clinical intensive care units (ICUs) refuse to deploy them autonomously. 
> 
> When I investigated *why*, I discovered a critical flaw called **Softmax Overconfidence**. Traditional neural networks are built to give a hard prediction no matter what. If an elderly patient rolls over in bed and their ECG electrode slips, creating jagged electrical noise, a standard deep learning model won't say *'I don't know'*. Instead, it will look at the noise and confidently predict *'Ventricular Tachycardia'* with 99% certainty!
>
> Furthermore, in real life, lethal arrhythmias like **Fusion beats** are extremely rare, while healthy normal beats are abundant. Standard models get biased towards normal beats and silently misclassify the dangerous ones. 
>
> That led to my central research question:  
> *Can we build an AI that not only classifies heartbeats with high precision, but is also self-aware enough to know when it is uncertain, isolate signal noise from model confusion, and safely refuse to guess when patient safety is on the line?*  
> That is the foundation of this project."

---

### 2. Where I Got the Dataset (MIT-BIH via PhysioNet)
> "To build a clinically valid system, I needed real human patient data rather than synthetic simulations.
>
> I sourced our data from **PhysioNet**, a public research repository maintained by **MIT and Beth Israel Hospital in Boston**, funded by the National Institutes of Health (NIH). Specifically, I utilized the **MIT-BIH Arrhythmia Database**, which has been the globally recognized gold-standard benchmark in cardiac research since 1980.
>
> **The dataset characteristics:**
> * It consists of **48 continuous, half-hour, 2-channel ambulatory Holter ECG recordings** taken from 47 individual subjects (25 men and 22 women).
> * The signals were digitized at a clinical sampling rate of **360 samples per second (360 Hz)** per channel, with 11-bit resolution over a 10 mV range.
> * Most importantly, **every single heartbeat in this database was manually annotated and cross-verified by two independent certified cardiologists**. 
> * For this project, I used **Lead MLII (Modified Limb Lead II)**, which is the standard lead used in telemetry because it aligns with the heart's electrical axis and makes the P-wave and QRS complex prominent."

---

### 3. Step-by-Step Data Engineering & Preprocessing Pipeline
> "Raw ECG signals from a moving patient are filled with physical noise. Before feeding them to any neural network, I designed a **multi-stage signal processing pipeline in Python using the `wfdb` (WaveForm DataBase) library**:
>
> **Step 1 — Noise Filtering (0.5 to 45 Hz Butterworth Bandpass):**  
> Raw recordings suffer from two major noise sources:
> 1. *Low-frequency baseline wander* (0.1–0.5 Hz) caused by patient breathing and chest movement.
> 2. *High-frequency noise* (>50 Hz) caused by electromyographic muscle tremor and 50/60 Hz power-line interference.  
> To clean this without distorting the heartbeat shape, I implemented a zero-phase **2nd-order Butterworth Bandpass Filter** with a passband between **0.5 Hz and 45 Hz**.
>
> **Step 2 — R-Peak Centering & Segmentation:**  
> Using the cardiologists' verified annotation locations, I extracted individual beats. To capture the full electrophysiological cycle (the P-wave, QRS complex, and T-wave), I extracted a **360-sample window** around each R-peak. That is exactly **1.0 second of cardiac data** (90 samples before the peak, and 270 samples after).
>
> **Step 3 — Z-Score Normalization:**  
> Different patients have different chest skin impedances and signal voltages. To prevent amplitude discrepancies from confusing the neural network, I applied **Z-score standardization** to every beat vector:  
> $$x_{norm} = \frac{x - \mu}{\sigma}$$  
> This ensures every beat has zero mean and unit variance.
>
> **Step 4 — Standard Clinical Class Mapping (AAMI EC57):**  
> The raw database contains over 15 granular rhythm codes. Following the clinical **ANSI/AAMI EC57 international standard**, I mapped all 109,446 beats into the **5 universal cardiac classes**:
> 1. **Class N (Normal / Bundle Branch Block):** 90,589 beats
> 2. **Class S (Supraventricular Ectopic):** 2,779 beats
> 3. **Class V (Ventricular Ectopic / Premature Contractions):** 7,236 beats
> 4. **Class F (Fusion of Ventricular & Normal):** 803 beats *(severe minority)*
> 5. **Class Q (Unknown / Paced):** 8,039 beats
>
> **Step 5 — Train/Test Stratified Split:**  
> I split the 109,446 beats into **87,558 training beats (80%)** and **21,888 independent test beats (20%)**, maintaining exact class ratios."

---

### 4. Detailed 1D-CNN Model Architecture
> "Now let me explain the model architecture. 
> 
> Many researchers convert ECG signals into 2D spectrogram images and pass them into heavy image networks like ResNet-50. I intentionally avoided this. Converting a 1D electrical signal into an image loses temporal phase information, adds unnecessary parameters, and slows down inference.
>
> Instead, I designed an optimized, high-speed **1D-Convolutional Neural Network (1D-CNN)** that processes the raw temporal signal directly:
>
> * **Input Layer:** Takes a 1-dimensional vector of size `(Batch_Size, 1, 360)`.
> * **Convolutional Block 1:**  
>   - 1D Convolution with **32 filters**, kernel size of 5, stride 1, padding 2.  
>   - Batch Normalization to stabilize activations.  
>   - ReLU non-linearity.  
>   - Max Pooling with kernel size 2 (reducing temporal dimension from 360 to 180).  
>   - Spatial Dropout of 0.2 to prevent co-adaptation.
> * **Convolutional Block 2:**  
>   - 1D Convolution with **64 filters**, kernel size 5.  
>   - Batch Normalization + ReLU.  
>   - Max Pooling with kernel size 2 (reducing dimension from 180 to 90).  
>   - Dropout of 0.2.
> * **Convolutional Block 3:**  
>   - 1D Convolution with **128 filters**, kernel size 3, capturing deep morphological features.  
>   - Batch Normalization + ReLU.  
>   - Max Pooling with kernel size 2 (reducing dimension from 90 to 45).  
>   - Dropout of 0.3.
> * **Classification Head:**  
>   - Flatten layer converting feature maps into a vector of size $45 \times 128 = 5,760$.  
>   - Fully Connected Dense Linear layer down to **128 hidden neurons** with ReLU and 0.4 Dropout.  
>   - Final Output Linear layer producing **5 raw unnormalized logits**, corresponding to classes N, S, V, F, and Q."

---

### 5. Step-by-Step Training of the 5-Member True Deep Ensemble
> "To eliminate single-model bias, I trained a **True Deep Ensemble of 5 independent 1D-CNN models**. Here is the exact training methodology:
>
> **Step 1 — Diversity via Seed Randomization:**  
> Rather than training one model or using simple cross-validation, I initialized 5 completely distinct networks with **5 distinct random seeds: 42, 101, 202, 303, and 404**. This forced each model’s weights to initialize in different regions of the non-convex loss surface and encounter mini-batches in different randomized orders.
>
> **Step 2 — Solving Class Imbalance via Class-Weighted Cross-Entropy:**  
> Because Normal beats outnumber Fusion beats by more than 100 to 1, standard cross-entropy would ignore Fusion beats. I calculated inverse-frequency class weights:  
> $$w_c = \frac{N_{total}}{K \cdot N_c}$$  
> This penalized the network heavily whenever it misclassified rare Fusion (F) or Supraventricular (S) beats.
>
> **Step 3 — Optimization Hyperparameters:**  
> - **Optimizer:** Adam optimizer with an initial learning rate of $\eta = 0.001$.  
> - **Batch Size:** 64 beats per mini-batch.  
> - **Epochs:** 5 full epochs per model (across 87,558 training beats, this is 1,368 gradient steps per epoch).
>
> **Step 4 — Training Loss Convergence:**  
> As documented in our **Figure 4.1**:
> - All 5 models started at an initial loss of approximately **0.39 to 0.41** in Epoch 1.  
> - They converged smoothly down to a final training loss between **0.127 and 0.142** in Epoch 5 (Ensemble mean loss = **0.1327**).  
> - No divergence or oscillation occurred, demonstrating exceptional optimization stability.
>
> **Step 5 — Model Checkpointing:**  
> Each converged model state was saved to disk (`model_seed_42.pth` through `model_seed_404.pth`), totaling 5 independent expert models ready for consensus voting."

---

### 6. Novel 3-Tier Uncertainty Pipeline & Temperature Calibration
> "Once the ensemble was trained, I integrated our **3-Tier Uncertainty Framework** to make the system clinically trustworthy:
>
> 1. **Tier 1 - Epistemic Model Uncertainty (MC Dropout):**  
>    During inference, I kept dropout active and executed $T=15$ stochastic forward passes across the ensemble. We compute the variance across predicted probabilities:  
>    $$\sigma^2_{MC} = \frac{1}{T}\sum_{t=1}^T (p_t - \bar{p})^2$$  
>    High variance means the model has never seen a waveform like this before.
>
> 2. **Tier 2 - Aleatoric Data Uncertainty (Predictive Entropy):**  
>    We calculate Shannon Entropy on the mean ensemble softmax probabilities:  
>    $$H(p) = -\sum_{c=1}^5 p_c \log_2(p_c)$$  
>    If probabilities are ambiguous (e.g. 50% Normal and 50% PVC), entropy spikes, flagging borderline beat shapes.
>
> 3. **Tier 3 - My Novel Innovation: Cluster-Based Entropy (CBE):**  
>    Standard Shannon entropy treats all confusion equally. But medically, confusing a Normal beat with a benign Supraventricular ectopic ($N \leftrightarrow S$) is acceptable, whereas confusing a Normal beat with a life-threatening Ventricular ectopic ($N \leftrightarrow V$) is dangerous!  
>    I grouped the 5 classes into **4 Clinical Risk Tiers**:  
>    - Tier 1: Normal $\{N\}$  
>    - Tier 2: Supraventricular $\{S\}$  
>    - Tier 3: Malignant Ventricular & Fusion $\{V, F\}$  
>    - Tier 4: Unknown / Paced $\{Q\}$  
>    CBE calculates entropy across these risk tiers:  
>    $$H_{cluster} = -\sum_{k=1}^4 P(Cluster_k) \log_2 P(Cluster_k)$$  
>    This suppresses false alarms on harmless sub-type variations, but strictly escalates life-threatening ventricular ambiguities!
>
> 4. **Post-Hoc Temperature Calibration:**  
>    Even an ensemble can produce overconfident raw probabilities. I optimized a post-hoc Temperature parameter $T = 1.48$ on validation logits via Negative Log-Likelihood. This dropped our **Expected Calibration Error (ECE)** from **0.084 down to 0.018**, ensuring probability values match true clinical correctness rates."

---

### 7. Quantitative Results & Live Clinical Dashboard Demo
> "Evaluating on the **21,888 unseen test beats**, our empirical results proved superior to single-model baselines:
>
> * **Overall Accuracy:** Reached **94.0%** (compared to 90.4% baseline).
> * **Macro Average F1:** Rose from **0.74 to 0.83**.
> * **Rare Fusion Beat F1 (Class F):** Jumped from **0.40 to 0.59**—an absolute improvement of **+19.0%**!
> * **Ventricular Sensitivity (Class V):** Achieved **98% Recall**, ensuring critical PVCs are not missed.
>
> **The Real-Time Clinical Dashboard:**  
> To demonstrate practical hospital deployment, I built a full-stack clinical interface using **Flask, HTML5 Canvas, and Web Audio**:
> * Streams continuous 360 Hz ECG with animated P-Q-R-S-T peaks and dynamic BPM.
> * Displays our **3D Uncertainty HUD** (Epistemic Variance, Predictive Entropy, and CBE gauges).
> * Synthesizes real-time cardiac audio beeps that shift tone during ventricular arrhythmias.
> * Includes an **Out-Of-Distribution (OOD) Safety Fallback**: If an electrode disconnects or noise spikes, the system halts autonomous prediction and alerts: *'CRITICAL CLINICAL SAFETY PROTOCOL TRIGGERED: Automated diagnosis suspended. Human Cardiologist Review Required.'*
> * A **50-Beat Holter Batch Engine** that automates bulk triage for busy cardiologists."

---

### 8. Conclusion & Future Directions
> "To conclude, this project bridges the gap between deep learning research and clinical trust. By moving away from overconfident black-box models and introducing a **5-member True Deep Ensemble**, **3-tier Uncertainty Quantification**, and **Temperature Calibration**, we provide cardiologists with a system that is accurate, calibrated, and humble enough to ask for human assistance when uncertain.
>
> For future research, I aim to deploy this model to low-cost **STM32 / ESP32 microcontrollers via ONNX Runtime** for wearable cardiac patches, and expand from single-lead to full 12-lead multi-channel ECG transformers.
>
> Thank you for your time. I am now delighted to answer your questions."

---

## 💡 Examiner Rapid-Fire Cheat Sheet (Quick Answers)

| Question Area | Examiner Might Ask | Your 10-Second Bulletproof Answer |
| :--- | :--- | :--- |
| **Origin of Idea** | *"Why did you do this instead of standard image classification?"* | *"Because 17.9M people die from CVD annually, and AI models fail in hospitals due to softmax overconfidence on noisy Holter data. I wanted an AI that knows when it does not know."* |
| **Dataset Source** | *"Where does this data come from?"* | *"PhysioNet MIT-BIH Arrhythmia Database from Beth Israel Hospital Boston & MIT; 48 patient records at 360 Hz with certified cardiologist beat annotations."* |
| **Why 1D-CNN?** | *"Why not 2D CNN or ResNet?"* | *"1D-CNN operates directly on the native 360-sample temporal voltage series without lossy spectrogram conversion, using 10x fewer parameters and running in under 5 ms."* |
| **Novelty (CBE)** | *"What makes Cluster-Based Entropy novel?"* | *"Standard entropy treats N vs S confusion the same as N vs V. Medically, N vs V is lethal. CBE clusters classes into 4 risk tiers so only life-threatening ambiguities sound alarms."* |
| **Ensemble Training** | *"Why 5 seeds instead of K-fold?"* | *"K-fold trains on different data subsets. Deep Ensembles train on the full dataset with randomized weight initializations (seeds 42 to 404), exploring distinct non-convex minima to capture epistemic variance."* |
| **Fusion Beats (+19%)**| *"Why did Fusion beats improve so much?"* | *"Fusion beats represent only 0.7% of the data. Inverse class weighting forced the gradient to penalize Fusion errors, while the 5-model consensus voting prevented the majority Normal class from overwhelming it."* |
| **Temperature $T=1.48$**| *"Does Temperature Scaling change accuracy?"* | *"No, because dividing logits by scalar $T$ is strictly monotonic; the argmax ranking remains unchanged. It strictly softens overconfidence, lowering ECE from 0.084 to 0.018."* |
