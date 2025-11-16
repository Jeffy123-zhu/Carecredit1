"""
Alzheimer's Disease Detection System - Web Application
Futuristic AI-Powered Medical Analysis Platform
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warm and caring CSS styling for elderly-friendly design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c5f7c;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: 1px;
        line-height: 1.2;
    }
    
    .subtitle {
        text-align: center;
        color: #5a8fa8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: -5px;
        font-style: italic;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 2px solid #d4e6f1;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(44, 95, 124, 0.1);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #5a8fa8 0%, #2c5f7c 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.9rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 3px 12px rgba(44, 95, 124, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #6ba0b8 0%, #3a6f8c 100%);
        box-shadow: 0 4px 16px rgba(44, 95, 124, 0.4);
        transform: translateY(-1px);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    .upload-area {
        background: #f0f7fa;
        border: 2px dashed #5a8fa8;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #5a8fa8;
        padding: 18px;
        border-radius: 8px;
        margin: 10px 0;
        color: #2c5f7c;
    }
    
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #ffb74d;
        padding: 18px;
        border-radius: 8px;
        margin: 10px 0;
        color: #5d4037;
    }
    
    .error-box {
        background: #ffebee;
        border-left: 4px solid #e57373;
        padding: 18px;
        border-radius: 8px;
        margin: 10px 0;
        color: #c62828;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #2c5f7c;
        font-weight: 600;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    
    .stSlider {
        color: #5a8fa8;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    .stMarkdown {
        line-height: 1.8;
    }
    
    .stMarkdown p {
        font-size: 1.05rem;
        color: #4a5568;
    }
    
    .caring-message {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e6f1 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #b8d4e3;
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Caring Technology for Early Detection and Support | SADS v3.0</p>', unsafe_allow_html=True)
st.markdown("---")

    # Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings & Configuration")
    
    # Dataset path configuration
    st.subheader("📁 Dataset Path")
    # Detect if running on Streamlit Cloud
    is_streamlit_cloud = (
        'streamlit.app' in os.environ.get('SERVER_NAME', '') or
        'STREAMLIT_SERVER_PORT' in os.environ or
        os.environ.get('STREAMLIT_SHARING_MODE') == 'public' or
        'share.streamlit.io' in os.environ.get('SERVER_NAME', '')
    )
    
    if is_streamlit_cloud:
        default_path = "./Datasets"
        st.info("🌐 **Streamlit Cloud Environment Detected**")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("💡 **Please use the file upload feature below to upload your data files.**")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        default_path = r"C:\Users\Administrator\Downloads\Datasets-20251115T200020Z-1-001\Datasets"
    
    dataset_path = st.text_input("Dataset Root Directory (Local Only)", value=default_path, disabled=is_streamlit_cloud, help="For local use only. On Streamlit Cloud, please use file upload instead.")
    
    # Data source selection (moved before file upload)
    st.markdown("---")
    st.subheader("📊 Data Source Selection")
    use_alz_variant = st.checkbox("Use ALZ_Variant Data", value=True)
    use_mri = st.checkbox("Use MRI Data", value=True)
    combine_datasets = st.checkbox("Combine Datasets", value=True)
    
    # File upload section
    st.markdown("---")
    st.subheader("📤 Upload Data Files")
    
    if is_streamlit_cloud:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("""
        **📤 File Upload Required**
        
        Streamlit Cloud cannot access your local files. Please upload your data files here to begin the analysis.
        
        **💡 Important:** 
        - Each uploader accepts **only ONE file**
        - Upload files **one at a time**
        - Only data files (.npz or .parquet) are accepted
        - Do NOT upload .md, .txt, or other text files
        - Your files are securely stored in this session
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("💡 **Local Environment:** You can use file paths from your computer or upload files directly here.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ALZ_Variant upload
    if use_alz_variant:
        st.markdown("**ALZ_Variant Data:**")
        uploaded_alz = st.file_uploader(
            "Choose file: preprocessed_alz_data.npz", 
            type=['npz'], 
            key='alz_uploader',
            accept_multiple_files=False,
            help="Upload ONE .npz file containing preprocessed ALZ_Variant data"
        )
        if uploaded_alz is not None:
            st.session_state.uploaded_alz = uploaded_alz
            st.success(f"✓ **Uploaded:** {uploaded_alz.name} ({uploaded_alz.size / 1024 / 1024:.2f} MB)")
        elif is_streamlit_cloud:
            st.caption("⚠️ Please upload the ALZ_Variant data file (.npz format only)")
    else:
        uploaded_alz = None
    
    # MRI uploads
    if use_mri:
        st.markdown("**MRI Training Data:**")
        uploaded_mri_train = st.file_uploader(
            "Choose file: train.parquet", 
            type=['parquet'], 
            key='mri_train_uploader',
            accept_multiple_files=False,
            help="Upload ONE .parquet file containing MRI training data"
        )
        if uploaded_mri_train is not None:
            st.session_state.uploaded_mri_train = uploaded_mri_train
            st.success(f"✓ **Uploaded:** {uploaded_mri_train.name} ({uploaded_mri_train.size / 1024 / 1024:.2f} MB)")
        elif is_streamlit_cloud:
            st.caption("⚠️ Please upload the MRI training data file (.parquet format only)")
        
        st.markdown("**MRI Test Data:**")
        uploaded_mri_test = st.file_uploader(
            "Choose file: test.parquet", 
            type=['parquet'], 
            key='mri_test_uploader',
            accept_multiple_files=False,
            help="Upload ONE .parquet file containing MRI test data"
        )
        if uploaded_mri_test is not None:
            st.session_state.uploaded_mri_test = uploaded_mri_test
            st.success(f"✓ **Uploaded:** {uploaded_mri_test.name} ({uploaded_mri_test.size / 1024 / 1024:.2f} MB)")
        elif is_streamlit_cloud:
            st.caption("⚠️ Please upload the MRI test data file (.parquet format only)")
    else:
        uploaded_mri_train = None
        uploaded_mri_test = None
    
    # Model training parameters
    st.markdown("---")
    st.subheader("🎯 TRAINING PARAMETERS")
    use_ensemble = st.checkbox("Use 4-Model Ensemble (Recommended)", value=True)
    epochs = st.slider("Training Epochs", 10, 50, 20)
    batch_size = st.slider("Batch Size", 8, 32, 16)
    
    # Run button
    st.markdown("---")
    run_analysis = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    # About information
    st.markdown("---")
    st.markdown("### 📖 About This System")
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("""
    **This system helps with early detection of Alzheimer's Disease through:**
    
    • **Genetic Analysis** - ALZ_Variant genetic variant data
    • **Brain Imaging** - MRI imaging data analysis
    • **Advanced AI** - 4-model ensemble learning for accurate results
    
    *Designed with care and compassion for patients and families.*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if run_analysis:
    # Progress display
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load modules
        status_text.text("📦 Loading modules...")
        progress_bar.progress(10)
        
        # Check if running on Streamlit Cloud
        is_streamlit_cloud = (
            'streamlit.app' in os.environ.get('SERVER_NAME', '') or
            'STREAMLIT_SERVER_PORT' in os.environ or
            os.environ.get('STREAMLIT_SHARING_MODE') == 'public' or
            'share.streamlit.io' in os.environ.get('SERVER_NAME', '')
        )
        
        # Set paths
        BASE_DATASET_PATH = dataset_path
        ALZ_VARIANT_PATH = os.path.join(BASE_DATASET_PATH, "ALZ_Variant")
        MRI_PATH = os.path.join(BASE_DATASET_PATH, "MRI")
        
        # Data loading
        status_text.text("📂 Loading dataset...")
        progress_bar.progress(20)
        
        X_train_final = None
        X_test_final = None
        y_train_final = None
        y_test_final = None
        data_source_info = []
        
        # Check path existence (local only)
        path_errors = []
        if not is_streamlit_cloud and not os.path.exists(BASE_DATASET_PATH):
            path_errors.append(f"Dataset root directory does not exist: {BASE_DATASET_PATH}")
        
        # Load ALZ_Variant data
        if use_alz_variant:
            alz_data = None
            
            # Priority: uploaded files (Streamlit Cloud)
            if is_streamlit_cloud:
                if 'uploaded_alz' in st.session_state and st.session_state.uploaded_alz is not None:
                    try:
                        import numpy as np
                        import io
                        alz_data = np.load(io.BytesIO(st.session_state.uploaded_alz.read()))
                        st.session_state.uploaded_alz.seek(0)
                    except Exception as e:
                        path_errors.append(f"ALZ_Variant upload file read failed: {str(e)}")
                else:
                    path_errors.append("Please upload ALZ_Variant data file (preprocessed_alz_data.npz)")
            else:
                # Local file system
                alz_npz_path = os.path.join(ALZ_VARIANT_PATH, "preprocessed_alz_data.npz")
                if os.path.exists(alz_npz_path):
                    try:
                        import numpy as np
                        alz_data = np.load(alz_npz_path)
                    except Exception as e:
                        path_errors.append(f"ALZ_Variant file read failed: {str(e)}")
                else:
                    path_errors.append(f"ALZ_Variant file does not exist: {alz_npz_path}")
            
            if alz_data is not None:
                try:
                    X_train_alz = alz_data['X_train']
                    X_test_alz = alz_data['X_test']
                    y_train_alz = alz_data['y_train']
                    y_test_alz = alz_data['y_test']
                    
                    # Convert to binary classification
                    if len(y_train_alz.shape) > 1:
                        y_train_alz_binary = (np.argmax(y_train_alz, axis=1) >= 7).astype(int)
                        y_test_alz_binary = (np.argmax(y_test_alz, axis=1) >= 7).astype(int)
                    else:
                        y_train_alz_binary = (y_train_alz > 0.5).astype(int)
                        y_test_alz_binary = (y_test_alz > 0.5).astype(int)
                    
                    X_train_alz_seq = np.stack([X_train_alz, X_train_alz * 0.95], axis=1)
                    X_test_alz_seq = np.stack([X_test_alz, X_test_alz * 0.95], axis=1)
                    
                    X_train_final = X_train_alz_seq
                    X_test_final = X_test_alz_seq
                    y_train_final = y_train_alz_binary
                    y_test_final = y_test_alz_binary
                    data_source_info.append("ALZ_Variant")
                    st.success(f"✓ ALZ_Variant data loaded successfully: {X_train_alz.shape[0]} samples")
                except Exception as e:
                    path_errors.append(f"ALZ_Variant data processing failed: {str(e)}")
        
        # Load MRI data
        if use_mri:
            mri_train = None
            mri_test = None
            
            # Priority: uploaded files (Streamlit Cloud)
            if is_streamlit_cloud:
                if 'uploaded_mri_train' in st.session_state and st.session_state.uploaded_mri_train is not None and \
                   'uploaded_mri_test' in st.session_state and st.session_state.uploaded_mri_test is not None:
                    try:
                        import io
                        mri_train = pd.read_parquet(io.BytesIO(st.session_state.uploaded_mri_train.read()))
                        mri_test = pd.read_parquet(io.BytesIO(st.session_state.uploaded_mri_test.read()))
                        st.session_state.uploaded_mri_train.seek(0)
                        st.session_state.uploaded_mri_test.seek(0)
                    except Exception as e:
                        path_errors.append(f"MRI upload file read failed: {str(e)}")
                else:
                    if 'uploaded_mri_train' not in st.session_state or st.session_state.uploaded_mri_train is None:
                        path_errors.append("Please upload MRI training data file (train.parquet)")
                    if 'uploaded_mri_test' not in st.session_state or st.session_state.uploaded_mri_test is None:
                        path_errors.append("Please upload MRI test data file (test.parquet)")
            else:
                # Local file system
                mri_train_path = os.path.join(MRI_PATH, "train.parquet")
                mri_test_path = os.path.join(MRI_PATH, "test.parquet")
                
                if os.path.exists(mri_train_path) and os.path.exists(mri_test_path):
                    try:
                        mri_train = pd.read_parquet(mri_train_path)
                        mri_test = pd.read_parquet(mri_test_path)
                    except Exception as e:
                        path_errors.append(f"MRI file read failed: {str(e)}")
                else:
                    if not os.path.exists(mri_train_path):
                        path_errors.append(f"MRI training file does not exist: {mri_train_path}")
                    if not os.path.exists(mri_test_path):
                        path_errors.append(f"MRI test file does not exist: {mri_test_path}")
            
            if mri_train is not None and mri_test is not None:
                try:
                    # Process MRI data
                    target_col = mri_train.columns[-1]
                    feature_cols_mri = [col for col in mri_train.columns if col != target_col]
                    
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='mean')
                    X_train_mri = imputer.fit_transform(mri_train[feature_cols_mri].values)
                    X_test_mri = imputer.transform(mri_test[feature_cols_mri].values)
                    
                    y_train_mri = mri_train[target_col].values
                    y_test_mri = mri_test[target_col].values
                    
                    if y_train_mri.dtype == object:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_train_mri = le.fit_transform(y_train_mri)
                        y_test_mri = le.transform(y_test_mri)
                    
                    if len(np.unique(y_train_mri)) > 2:
                        y_train_mri = (y_train_mri == np.max(y_train_mri)).astype(int)
                        y_test_mri = (y_test_mri == np.max(y_test_mri)).astype(int)
                    
                    X_train_mri_seq = np.stack([X_train_mri, X_train_mri * 0.95], axis=1)
                    X_test_mri_seq = np.stack([X_test_mri, X_test_mri * 0.95], axis=1)
                    
                    if combine_datasets and X_train_final is not None:
                        min_features = min(X_train_final.shape[2], X_train_mri_seq.shape[2])
                        X_train_final = np.concatenate([
                            X_train_final[:, :, :min_features],
                            X_train_mri_seq[:, :, :min_features]
                        ], axis=0)
                        X_test_final = np.concatenate([
                            X_test_final[:, :, :min_features],
                            X_test_mri_seq[:, :, :min_features]
                        ], axis=0)
                        y_train_final = np.concatenate([y_train_final, y_train_mri])
                        y_test_final = np.concatenate([y_test_final, y_test_mri])
                        data_source_info.append("MRI")
                        st.success(f"✓ MRI data loaded successfully: {mri_train.shape[0]} samples")
                    elif X_train_final is None:
                        X_train_final = X_train_mri_seq
                        X_test_final = X_test_mri_seq
                        y_train_final = y_train_mri
                        y_test_final = y_test_mri
                        data_source_info.append("MRI")
                        st.success(f"✓ MRI data loaded successfully: {mri_train.shape[0]} samples")
                except Exception as e:
                    path_errors.append(f"MRI data processing failed: {str(e)}")
        
        # Display detailed error information
        if X_train_final is None:
            st.error("❌ **Failed to load data!**")
            
            if is_streamlit_cloud:
                st.markdown("### 📤 File Upload Required")
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.info("""
                **We need your data files to begin the analysis.**
                
                **Here's what to do:**
                1. Look at the sidebar on the left → Find the "UPLOAD DATA FILES" section
                2. Upload the files you need based on your selection:
                   - If using **ALZ_Variant**: Upload `preprocessed_alz_data.npz`
                   - If using **MRI**: Upload both `train.parquet` and `test.parquet`
                3. Wait for the green checkmark confirmation
                4. Come back here and click "START ANALYSIS" again
                
                **💡 Remember:** Your files are safely stored in this session. If you refresh the page, you'll need to upload them again.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show what files are needed
                st.markdown("### 📋 Files Needed:")
                if use_alz_variant:
                    if 'uploaded_alz' not in st.session_state or st.session_state.uploaded_alz is None:
                        st.error("❌ **Missing:** ALZ_Variant data (preprocessed_alz_data.npz)")
                    else:
                        st.success(f"✓ **Uploaded:** {st.session_state.uploaded_alz.name}")
                
                if use_mri:
                    if 'uploaded_mri_train' not in st.session_state or st.session_state.uploaded_mri_train is None:
                        st.error("❌ **Missing:** MRI training data (train.parquet)")
                    else:
                        st.success(f"✓ **Uploaded:** {st.session_state.uploaded_mri_train.name}")
                    
                    if 'uploaded_mri_test' not in st.session_state or st.session_state.uploaded_mri_test is None:
                        st.error("❌ **Missing:** MRI test data (test.parquet)")
                    else:
                        st.success(f"✓ **Uploaded:** {st.session_state.uploaded_mri_test.name}")
            else:
                st.markdown("### Error Details:")
                for error in path_errors:
                    st.markdown(f'<div class="error-box">• {error}</div>', unsafe_allow_html=True)
                
                st.markdown("### 💡 Solution:")
                st.info(f"""
                1. **Check dataset path**: Current path is `{BASE_DATASET_PATH}`
                
                2. **Ensure files exist**:
                   - ALZ_Variant: `{ALZ_VARIANT_PATH}/preprocessed_alz_data.npz`
                   - MRI: `{MRI_PATH}/train.parquet` and `{MRI_PATH}/test.parquet`
                
                3. **Modify path in sidebar**: If dataset is in a different location, update the path in the sidebar
                
                4. **Check data source selection**: Ensure at least one data source is selected (ALZ_Variant or MRI)
                
                5. **Alternative**: Use the file upload feature in the sidebar to upload files directly
                """)
            st.stop()
        
        status_text.text("🔧 Preprocessing data...")
        progress_bar.progress(40)
        
        # Data standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_2d = X_train_final.reshape(-1, X_train_final.shape[-1])
        X_test_2d = X_test_final.reshape(-1, X_test_final.shape[-1])
        scaler.fit(X_train_2d)
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train_final.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test_final.shape)
        
        # Display data information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", f"{X_train_final.shape[0]:,}")
        with col2:
            st.metric("Test Samples", f"{X_test_final.shape[0]:,}")
        with col3:
            st.metric("Features", X_train_final.shape[2])
        with col4:
            st.metric("Data Sources", ", ".join(data_source_info))
        
        status_text.text("🏗️ Building models...")
        progress_bar.progress(50)
        
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        n_biomarkers = X_train_scaled.shape[2]
        
        if use_ensemble:
            # Build 4-model ensemble
            def build_lstm_model():
                return keras.Sequential([
                    layers.LSTM(32, activation='relu', return_sequences=True, 
                               input_shape=(2, n_biomarkers)),
                    layers.Dropout(0.3),
                    layers.LSTM(16, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
            
            def build_cnn_model():
                return keras.Sequential([
                    layers.Conv1D(32, kernel_size=1, activation='relu', 
                                 input_shape=(2, n_biomarkers)),
                    layers.MaxPooling1D(pool_size=1),
                    layers.Conv1D(16, kernel_size=1, activation='relu'),
                    layers.Flatten(),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
            
            def build_attention_model():
                inputs = keras.Input(shape=(2, n_biomarkers))
                attention = layers.MultiHeadAttention(num_heads=4, key_dim=8)(inputs, inputs)
                attention = layers.Flatten()(attention)
                x = layers.Dense(32, activation='relu')(attention)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(16, activation='relu')(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                return keras.Model(inputs=inputs, outputs=outputs)
            
            def build_hybrid_model():
                inputs = keras.Input(shape=(2, n_biomarkers))
                lstm = layers.LSTM(24, activation='relu', return_sequences=False)(inputs)
                cnn = layers.Conv1D(24, kernel_size=1, activation='relu')(inputs)
                cnn = layers.Flatten()(cnn)
                merged = layers.Concatenate()([lstm, cnn])
                x = layers.Dense(32, activation='relu')(merged)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(16, activation='relu')(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                return keras.Model(inputs=inputs, outputs=outputs)
            
            models = {
                'LSTM': build_lstm_model(),
                'CNN': build_cnn_model(),
                'Attention': build_attention_model(),
                'Hybrid': build_hybrid_model()
            }
            
            for name, model in models.items():
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            status_text.text("🎓 Training 4-model ensemble...")
            progress_bar.progress(60)
            
            # Train all models
            model_histories = {}
            training_placeholder = st.empty()
            
            for model_name, model in models.items():
                with training_placeholder.container():
                    st.info(f"Training {model_name} model...")
                history = model.fit(
                    X_train_scaled, y_train_final,
                    validation_split=0.2,
                    epochs=min(epochs, 20),
                    batch_size=batch_size,
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                    )],
                    verbose=0
                )
                model_histories[model_name] = history
            
            training_placeholder.empty()
            
            status_text.text("📊 Evaluating ensemble models...")
            progress_bar.progress(80)
            
            # Ensemble prediction
            ensemble_preds = []
            for model in models.values():
                pred = model.predict(X_test_scaled, verbose=0).flatten()
                ensemble_preds.append(pred)
            
            y_pred_proba = np.mean(ensemble_preds, axis=0)
            y_pred = (y_pred_proba > 0.5).astype(int)
            history = model_histories['LSTM']
            
        else:
            # Single model version
            model = keras.Sequential([
                layers.LSTM(32, activation='relu', return_sequences=True, 
                           input_shape=(2, n_biomarkers)),
                layers.Dropout(0.3),
                layers.LSTM(16, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            status_text.text("🎓 Training model...")
            progress_bar.progress(60)
            
            with st.spinner("Training in progress, please wait..."):
                history = model.fit(
                    X_train_scaled, y_train_final,
                    validation_split=0.2,
                    epochs=min(epochs, 20),
                    batch_size=batch_size,
                    verbose=0
                )
            
            status_text.text("📊 Evaluating model...")
            progress_bar.progress(80)
            
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            models = {'Single Model': model}
            ensemble_preds = [y_pred_proba]
        
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
        
        auc = roc_auc_score(y_test_final, y_pred_proba)
        accuracy = accuracy_score(y_test_final, y_pred)
        f1 = f1_score(y_test_final, y_pred)
        cm = confusion_matrix(y_test_final, y_pred)
        
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        st.markdown("## 📈 Analysis Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("AUC-ROC", f"{auc:.4f}", delta=None)
        with col2:
            st.metric("Accuracy", f"{accuracy:.4f}", delta=None)
        with col3:
            st.metric("F1-Score", f"{f1:.4f}", delta=None)
        with col4:
            st.metric("Sensitivity", f"{sensitivity:.4f}", delta=None)
        with col5:
            st.metric("Specificity", f"{specificity:.4f}", delta=None)
        
        # Display individual model performance (if ensemble)
        if use_ensemble and len(ensemble_preds) > 1:
            st.markdown("## 🔍 Individual Model Performance")
            model_aucs = {}
            for model_name, pred in zip(models.keys(), ensemble_preds):
                model_auc = roc_auc_score(y_test_final, pred)
                model_aucs[model_name] = model_auc
            
            model_df = pd.DataFrame({
                'Model': list(model_aucs.keys()),
                'AUC-ROC': list(model_aucs.values())
            })
            st.dataframe(model_df, use_container_width=True)
        
        # Visualization
        st.markdown("## 📊 Visualization Results")
        
        if use_ensemble and len(ensemble_preds) > 1:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC={auc:.4f}')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 0].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False, square=True)
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # Prediction distribution
        axes[1, 0].hist(y_pred_proba[y_test_final==0], bins=15, alpha=0.6, label='Normal', color='green')
        axes[1, 0].hist(y_pred_proba[y_test_final==1], bins=15, alpha=0.6, label='Alzheimer\'s', color='red')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics bar chart
        if use_ensemble and len(ensemble_preds) > 1:
            metrics = ['AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity']
            values = [auc, accuracy, f1, sensitivity, specificity]
            colors = ['#4ECDC4' if v > 0.8 else '#FF6B6B' for v in values]
            axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Overall Performance')
            axes[1, 1].set_ylim([0.5, 1.0])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Model comparison
            model_aucs = [roc_auc_score(y_test_final, pred) for pred in ensemble_preds]
            colors_models = ['#4ECDC4' if auc_val == max(model_aucs) else '#FF6B6B' for auc_val in model_aucs]
            axes[1, 2].bar(models.keys(), model_aucs, color=colors_models, alpha=0.7, edgecolor='black')
            axes[1, 2].set_ylabel('AUC-ROC')
            axes[1, 2].set_title('Individual Model Performance')
            axes[1, 2].set_ylim([0.7, 1.0])
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        else:
            metrics = ['AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity']
            values = [auc, accuracy, f1, sensitivity, specificity]
            colors = ['#4ECDC4' if v > 0.8 else '#FF6B6B' for v in values]
            axes[1, 1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Overall Performance')
            axes[1, 1].set_ylim([0.5, 1.0])
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Training history
        st.markdown("## 📉 Training History")
        history_df = pd.DataFrame(history.history)
        st.line_chart(history_df[['loss', 'val_loss']])
        
        # Clinical prediction examples
        st.markdown("## 🏥 Patient Risk Assessment")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("""
        **Understanding the Results:**
        
        The risk probability indicates the likelihood of Alzheimer's Disease based on the analysis. 
        These results should be reviewed with a qualified healthcare professional for proper interpretation and care planning.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        for i in range(min(5, len(X_test_scaled))):
            prob = y_pred_proba[i]
            if prob > 0.75:
                risk = "🔴 Very High Risk"
                risk_color = "#c62828"
            elif prob > 0.6:
                risk = "🟠 High Risk"
                risk_color = "#f57c00"
            elif prob > 0.4:
                risk = "🟡 Moderate Risk"
                risk_color = "#fbc02d"
            else:
                risk = "🟢 Low Risk"
                risk_color = "#388e3c"
            
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color}; margin: 10px 0;'>
                <strong>Patient {i+1}</strong><br>
                Risk Probability: <strong>{prob:.1%}</strong> | {risk}
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        st.exception(e)

else:
    # Welcome page
    st.markdown('<div class="caring-message">', unsafe_allow_html=True)
    st.markdown("""
    ## 👋 Welcome to the Alzheimer's Disease Detection System
    
    *A compassionate tool designed to support early detection and care planning*
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This system uses advanced technology to help identify early signs of Alzheimer's Disease by analyzing genetic and brain imaging data. Our goal is to provide support and clarity for patients and their families.
    
    ### ✨ How We Help
    
    **1. Comprehensive Analysis**
       - Genetic variant analysis (ALZ_Variant data)
       - Brain imaging assessment (MRI data)
       - Automatic data processing for accuracy
    
    **2. Advanced AI Technology**
       - Multiple learning models working together
       - Careful validation and testing
       - Reliable and trustworthy results
    
    **3. Clear Results**
       - Easy-to-understand visualizations
       - Detailed performance metrics
       - Risk assessment for each patient
    
    **4. Supportive Information**
       - Patient risk probability assessment
       - Visual charts and graphs
       - Helpful guidance for next steps
    
    ### 🚀 Getting Started
    
    **Step 1:** Upload your data files in the sidebar (if using Streamlit Cloud)
    
    **Step 2:** Choose which data sources to analyze
    
    **Step 3:** Adjust settings if needed (default settings work well)
    
    **Step 4:** Click "START ANALYSIS" to begin
    
    ### 📊 What You'll Need
    
    - **ALZ_Variant Data**: `preprocessed_alz_data.npz` file
    - **MRI Data**: `train.parquet` and `test.parquet` files
    
    ### 💙 Important Reminders
    
    - This tool is designed to assist healthcare professionals
    - Results should be reviewed with a qualified medical professional
    - Early detection can help with planning and support
    - We're here to help, not to replace professional medical care
    
    ### 🌟 Our Commitment
    
    We understand that dealing with Alzheimer's Disease can be challenging. This system is designed with care, compassion, and respect for patients and families. Our technology is here to support you on this journey.
    """)
    
    # Display dataset information
    st.markdown("### 📁 Dataset Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **ALZ_Variant Data**
        - Format: NPZ (NumPy compressed)
        - Training set: 5076 samples × 130 features
        - Test set: 1270 samples × 130 features
        - Labels: 9-class (converted to binary)
        """)
    
    with info_col2:
        st.markdown("""
        **MRI Data**
        - Format: Parquet (columnar storage)
        - Includes training and test sets
        - Imaging-related feature data
        - Suitable for big data analysis
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #5a8fa8; font-family: Inter, sans-serif; padding: 20px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;'>"
    "<p style='font-size: 0.95rem; margin: 5px 0;'><strong>Alzheimer's Disease Detection System (SADS v3.0)</strong></p>"
    "<p style='font-size: 0.85rem; margin: 5px 0; color: #7a9fb0;'>Designed with care and compassion</p>"
    "<p style='font-size: 0.8rem; margin: 5px 0; color: #9ab5c2;'>© 2025 | Powered by Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
