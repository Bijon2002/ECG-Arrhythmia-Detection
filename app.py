import sys
import os
import time
import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Connect to our 'src' folder so we can import our AI Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from model import ECGModel

# File Paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ecg_model_v1.pth')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data', 'samples', 'normal_sample.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Standard AAMI Medical Classes
CLASS_MAPPING = {
    0: ("Normal Beat (N)", "🟢"),
    1: ("Supraventricular Ectopic (S)", "🟡"),
    2: ("Ventricular Ectopic (V)", "🔴"),
    3: ("Fusion Beat (F)", "🟠"),
    4: ("Unknown Beat (Q)", "⚪")
}

st.set_page_config(page_title="ECG Arrhythmia AI", page_icon="🫀", layout="centered")

@st.cache_resource
def load_ai_model():
    """Loads the heavy PyTorch brain once and keeps it in memory."""
    model = ECGModel(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model

def enable_dropout(model):
    """Forces MC Dropout ON for uncertainty testing."""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

def run_ai_diagnosis(model, signal_window):
    """Runs the AI on a 360-sample window and returns UI elements."""
    tensor_signal = torch.tensor(signal_window, dtype=torch.float32).view(1, 1, 360)
    
    enable_dropout(model)
    mc_iterations = 5 # Reduced to 5 for speed during live animation
    mc_predictions = []
    
    with torch.no_grad():
        for _ in range(mc_iterations):
            logits = model(tensor_signal)
            probs = torch.softmax(logits, dim=1)
            mc_predictions.append(probs.numpy())
            
    mc_predictions = np.vstack(mc_predictions)
    mean_probs = np.mean(mc_predictions, axis=0)
    variance = np.var(mc_predictions, axis=0)
    
    predicted_class = int(np.argmax(mean_probs))
    confidence = mean_probs[predicted_class] * 100
    max_variance = np.max(variance)
    
    diagnosis_text, icon = CLASS_MAPPING[predicted_class]
    
    return diagnosis_text, icon, confidence, max_variance

def plot_ecg(signal_window):
    """Draws a beautiful neon medical monitor graph."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal_window, color='#00ff00', linewidth=2) # Neon green like a hospital monitor
    ax.set_facecolor('black') # Black background
    fig.patch.set_facecolor('black')
    
    # We remove the axes to make it look like a pure medical monitor
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    
    # Lock the y-axis so the graph doesn't jump up and down wildly while scrolling
    ax.set_ylim([-3.0, 3.0]) 
    return fig

def main():
    st.title("🫀 ECG Arrhythmia AI Diagnostic Tool")
    st.markdown("Upload a static ECG heartbeat or start the Live Monitor to simulate real-time AI diagnosis.")
    
    model = load_ai_model()
    
    st.sidebar.header("Data Input")
    use_sample = st.sidebar.button("📄 Load Static Sample (CSV)")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (1 column, 360 rows)", type=["csv"])
    
    st.sidebar.divider()
    
    # Our brand new Live Stream button!
    start_live = st.sidebar.button("📡 Start Live ECG Stream")
    
    # -----------------------------------------
    # STATIC MODE
    # -----------------------------------------
    if use_sample or uploaded_file is not None:
        signal = None
        if use_sample:
            df = pd.read_csv(SAMPLE_PATH, header=None)
            signal = df.values.flatten()
            st.sidebar.success("Static sample loaded!")
        else:
            df = pd.read_csv(uploaded_file, header=None)
            signal = df.values.flatten()
            if len(signal) != 360:
                st.sidebar.error("File must contain exactly 360 samples.")
                signal = None
                
        if signal is not None:
            st.subheader("Patient ECG Signal")
            fig = plot_ecg(signal)
            st.pyplot(fig)
            
            st.divider()
            st.subheader("AI Diagnosis")
            diag_text, icon, conf, max_var = run_ai_diagnosis(model, signal)
            
            st.markdown(f"### {icon} {diag_text}")
            st.progress(int(conf))
            st.write(f"**Confidence:** {conf:.2f}%")
            
            if max_var > 0.02:
                st.warning("⚠️ **HIGH UNCERTAINTY!** Please send to Cardiologist.")
            else:
                st.success("✅ **HIGH CERTAINTY.**")

    # -----------------------------------------
    # LIVE STREAM MODE
    # -----------------------------------------
    elif start_live:
        st.subheader("📡 Live Patient Monitor")
        
        # Load test data and stitch 30 random heartbeats together into a long continuous line
        test_data = np.load(os.path.join(PROCESSED_DIR, 'test_data.npz'))
        X = test_data['X']
        random_indices = np.random.choice(len(X), size=30, replace=False)
        continuous_signal = np.concatenate(X[random_indices])
        
        # Create placeholders that we can overwrite dynamically over and over
        graph_placeholder = st.empty()
        diagnosis_placeholder = st.empty()
        warning_placeholder = st.empty()
        
        window_size = 360
        step_size = 36 # Move forward by 0.1 seconds (36 samples) each frame
        
        # Slide the window across the continuous signal
        for start_idx in range(0, len(continuous_signal) - window_size, step_size):
            end_idx = start_idx + window_size
            current_window = continuous_signal[start_idx:end_idx]
            
            # Draw updated graph
            fig = plot_ecg(current_window)
            graph_placeholder.pyplot(fig)
            plt.close(fig) # Prevent memory leak
            
            # Run AI instantly
            diag_text, icon, conf, max_var = run_ai_diagnosis(model, current_window)
            
            # Build Diagnosis UI dynamically
            diag_html = f"<h3>{icon} {diag_text} (Conf: {conf:.1f}%)</h3>"
            diagnosis_placeholder.markdown(diag_html, unsafe_allow_html=True)
            
            if max_var > 0.02:
                warning_placeholder.warning("⚠️ **HIGH UNCERTAINTY!** Cardiologist review required.")
            else:
                warning_placeholder.empty() # Clear the warning if it's safe
                
            time.sleep(0.05) # Control the speed of the scrolling monitor
            
        st.success("Live Stream completed. Click the button to run it again!")

if __name__ == "__main__":
    main()
