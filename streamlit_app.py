import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition LSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .emotion-card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Emotion configurations
EMOTION_CONFIG = {
    'depressed': {'color': '#8B4789', 'emoji': '😔', 'description': 'Feeling down or hopeless'},
    'sad': {'color': '#4A90E2', 'emoji': '😢', 'description': 'Feeling unhappy or sorrowful'},
    'neutral': {'color': '#95A5A6', 'emoji': '😐', 'description': 'Neither positive nor negative'},
    'good': {'color': '#52C41A', 'emoji': '🙂', 'description': 'Feeling pleasant or satisfied'},
    'happy': {'color': '#FFA500', 'emoji': '😊', 'description': 'Feeling joyful or content'},
    'excited': {'color': '#FF6B6B', 'emoji': '🤩', 'description': 'Feeling enthusiastic or thrilled'}
}

# Default paths
DEFAULT_MODEL_PATH = "./models/model.keras"
DEFAULT_TOKENIZER_PATH = "./models/tokenizer.json"
DEFAULT_CONFIG_PATH = "./models/config.json"

@st.cache_resource
def load_emotion_model(model_path, tokenizer_path, config_path):
    """Load the trained LSTM model, tokenizer, and configuration"""
    try:
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None, None, None
        if not os.path.exists(tokenizer_path):
            st.error(f"❌ Tokenizer file not found: {tokenizer_path}")
            return None, None, None
        if not os.path.exists(config_path):
            st.error(f"❌ Config file not found: {config_path}")
            return None, None, None
        
        # Load model
        model = load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json_string = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json_string)
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return model, tokenizer, config
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def predict_emotion(text, model, tokenizer, config):
    """Predict emotion for a single text"""
    if not text.strip():
        return None
    
    # Preprocess text
    max_len = config['max_len']
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    predictions = model.predict(padded, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])
    
    # Get emotion label
    emotion_labels = config['emotion_labels']
    emotion = emotion_labels[predicted_class]
    
    # Create probability dictionary
    all_probabilities = {emotion_labels[i]: float(predictions[i]) 
                        for i in range(len(predictions))}
    
    result = {
        'text': text,
        'predicted_class': predicted_class,
        'emotion': emotion,
        'confidence': confidence,
        'all_probabilities': all_probabilities,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result

def create_probability_chart(probabilities, predicted_emotion):
    """Create interactive probability bar chart"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    colors = [EMOTION_CONFIG[em]['color'] if em == predicted_emotion 
              else '#D3D3D3' for em in emotions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probs,
            marker=dict(
                color=colors,
                line=dict(color='#333', width=2)
            ),
            text=[f'{p:.2%}' for p in probs],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_confidence_gauge(confidence, emotion):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Confidence Level<br><span style='font-size:0.8em'>{emotion.upper()}</span>"},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': EMOTION_CONFIG[emotion]['color']},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# App Header
st.title("🧠 Emotion Recognition using LSTM")
st.markdown("### Analyze emotions in text using deep learning")
st.markdown("Models are'nt perfect, results may vary based on input quality.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model paths section with expander
    with st.expander("📁 Model Paths (Optional: for local deployement)", expanded=False):
        st.caption("Default paths are already set. Only change if your files are in different locations.")
        model_path = st.text_input(
            "Model Path", 
            value=DEFAULT_MODEL_PATH,
            help="Path to the model.keras file"
        )
        tokenizer_path = st.text_input(
            "Tokenizer Path", 
            value=DEFAULT_TOKENIZER_PATH,
            help="Path to the tokenizer.json file"
        )
        config_path = st.text_input(
            "Config Path", 
            value=DEFAULT_CONFIG_PATH,
            help="Path to the config.json file"
        )
    
    # Auto-load model on first run
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading model..."):
            model, tokenizer, config = load_emotion_model(
                DEFAULT_MODEL_PATH,
                DEFAULT_TOKENIZER_PATH,
                DEFAULT_CONFIG_PATH
            )
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.config = config
                st.session_state.model_loaded = True
    
    # Manual reload button
    if st.button("🔄 Reload Model", help="Reload the model with current paths"):
        with st.spinner("Loading model..."):
            model, tokenizer, config = load_emotion_model(
                model_path, tokenizer_path, config_path
            )
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.config = config
                st.session_state.model_loaded = True
                st.success("✅ Model reloaded successfully!")
    
    # Display model info if loaded
    if st.session_state.model_loaded:
        st.success("✅ Model loaded")
        with st.expander("ℹ️ Model Info"):
            config = st.session_state.config
            st.markdown(f"""
            **Configuration:**
            - Max Length: `{config['max_len']}`
            - Vocab Size: `{config['vocab_size']:,}`
            - Classes: `{config['num_classes']}`
            - Emotions: {', '.join(config['emotion_labels'])}
            """)
    
    st.markdown("---")
    
    # Display emotion legend
    st.subheader("📊 Emotion Legend")
    for emotion, conf in EMOTION_CONFIG.items():
        st.markdown(f"{conf['emoji']} **{emotion.capitalize()}**")
        st.caption(conf['description'])
    
    st.markdown("---")
    
    # History controls
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
if not st.session_state.model_loaded:
    st.error("⚠️ Model could not be loaded. Please check the following:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Required Files")
        st.markdown("""
        Make sure these files exist in the `./models/` directory:
        
        1. ✅ `model.keras` - Trained LSTM model
        2. ✅ `tokenizer.json` - Fitted tokenizer
        3. ✅ `config.json` - Model configuration
        """)
        
        # Check file existence
        st.subheader("🔍 File Status")
        files_to_check = [
            (DEFAULT_MODEL_PATH, "Model"),
            (DEFAULT_TOKENIZER_PATH, "Tokenizer"),
            (DEFAULT_CONFIG_PATH, "Config")
        ]
        
        for file_path, file_name in files_to_check:
            if os.path.exists(file_path):
                st.success(f"✅ {file_name}: Found")
            else:
                st.error(f"❌ {file_name}: Not found at `{file_path}`")
    
    with col2:
        st.subheader("📝 About This App")
        st.markdown("""
        This application uses a Long Short-Term Memory (LSTM) neural network to classify emotions in text.
        
        **Supported Emotions:**
        - 😔 Depressed
        - 😢 Sad
        - 😐 Neutral
        - 🙂 Good
        - 😊 Happy
        - 🤩 Excited
        
        **Model Architecture:**
        - Embedding Layer (100 dimensions)
        - LSTM Layer (64 units)
        - Dropout (0.5)
        - Dense Layer (32 units, ReLU)
        - Dropout (0.5)
        - Output Layer (6 classes, Softmax)
        """)
        
        st.info("""
        💡 **Tip:** If files are in a different location, 
        expand the 'Model Paths' section in the sidebar 
        and update the paths accordingly.
        """)
else:
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text for Analysis")
        text_input = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Example: I'm so excited about my new project! This is going to be amazing!",
            help="Enter any text to analyze its emotional content"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🧹 Clear", use_container_width=True)
            if clear_btn:
                text_input = ""
                st.rerun()
    
    with col2:
        st.subheader("📊 Quick Examples")
        examples = {
            "Depressed": "I feel completely hopeless and lost.",
            "Sad": "I'm feeling down today.",
            "Neutral": "It's just another day.",
            "Good": "Things are going well.",
            "Happy": "I'm having a great time!",
            "Excited": "This is absolutely amazing!"
        }
        
        for emotion, example in examples.items():
            if st.button(f"{EMOTION_CONFIG[emotion.lower()]['emoji']} {emotion}", 
                        key=f"ex_{emotion}",
                        use_container_width=True):
                text_input = example
                st.rerun()
    
    # Perform prediction
    if analyze_btn and text_input:
        with st.spinner("🔄 Analyzing emotion..."):
            result = predict_emotion(
                text_input,
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.config
            )
            
            if result:
                # Add to history
                st.session_state.history.insert(0, result)
                
                st.markdown("---")
                
                # Display results
                st.subheader("🎯 Analysis Results")
                
                # Main emotion display
                emotion = result['emotion']
                confidence = result['confidence']
                emoji = EMOTION_CONFIG[emotion]['emoji']
                color = EMOTION_CONFIG[emotion]['color']
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                            padding: 2rem; border-radius: 15px; border-left: 5px solid {color};'>
                    <h1 style='margin:0; color:{color};'>{emoji} {emotion.upper()}</h1>
                    <h3 style='margin:0.5rem 0 0 0;'>Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Charts
                col_chart1, col_chart2 = st.columns([2, 1])
                
                with col_chart1:
                    fig_prob = create_probability_chart(
                        result['all_probabilities'],
                        emotion
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                with col_chart2:
                    fig_gauge = create_confidence_gauge(confidence, emotion)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Detailed probabilities
                with st.expander("📈 Detailed Probability Breakdown"):
                    sorted_probs = sorted(
                        result['all_probabilities'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for emo, prob in sorted_probs:
                        col_emo, col_bar, col_pct = st.columns([1, 3, 1])
                        with col_emo:
                            st.markdown(f"{EMOTION_CONFIG[emo]['emoji']} **{emo.capitalize()}**")
                        with col_bar:
                            st.progress(prob)
                        with col_pct:
                            st.markdown(f"**{prob:.2%}**")
    
    # History section
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Analysis History")
        
        # Create dataframe
        history_data = []
        for item in st.session_state.history[:10]:  # Show last 10
            history_data.append({
                'Timestamp': item['timestamp'],
                'Text': item['text'][:50] + '...' if len(item['text']) > 50 else item['text'],
                'Emotion': f"{EMOTION_CONFIG[item['emotion']]['emoji']} {item['emotion'].capitalize()}",
                'Confidence': f"{item['confidence']:.2%}"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # Emotion distribution
        with st.expander("📊 Emotion Distribution in History"):
            emotion_counts = {}
            for item in st.session_state.history:
                emotion = item['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            fig_dist = px.pie(
                values=list(emotion_counts.values()),
                names=list(emotion_counts.keys()),
                title="Distribution of Emotions in History",
                color_discrete_map={k: v['color'] for k, v in EMOTION_CONFIG.items()}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with Streamlit 🎈 | Powered by TensorFlow/Keras 🧠</p>
    <p><small>LSTM Emotion Recognition Model</small></p>
</div>
""", unsafe_allow_html=True)