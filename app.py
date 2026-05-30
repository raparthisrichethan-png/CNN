import os
import warnings

# Suppress TensorFlow warnings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
warnings.filterwarnings('ignore')

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Additional TensorFlow configuration
tf.get_logger().setLevel('ERROR')

st.set_page_config(page_title="CNN - Cat vs Dog Classifier", layout="wide")

st.title("🐱🐶 Cat vs Dog Classifier - CNN")
st.markdown("---")

# Sidebar for navigation
menu = st.sidebar.radio(
    "Select Option",
    ["🏠 Home", "📊 Train Model", "🔮 Predict", "📈 Model Info"]
)

# Dataset path
dataset_path = "PetImages"

if menu == "🏠 Home":
    st.header("Welcome to CNN Classifier")
    st.write("""
    This application uses a Convolutional Neural Network (CNN) to classify images as either:
    - 🐱 **Cats**
    - 🐶 **Dogs**
    
    ### Features:
    - Train a CNN model on your dataset
    - Make predictions on new images
    - View model architecture and performance
    
    ### How to use:
    1. **Train Model**: Go to the 'Train Model' section to train the CNN
    2. **Predict**: Upload an image to get predictions
    3. **Model Info**: View the model architecture and training history
    """)

elif menu == "📊 Train Model":
    st.header("Train CNN Model")
    
    if not os.path.exists(dataset_path):
        st.error(f"❌ Dataset path not found: {dataset_path}")
        st.info("Please make sure the 'PetImages' folder exists in the same directory as this app.")
    else:
        st.success(f"✅ Dataset found at: {dataset_path}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Number of Epochs", min_value=1, max_value=20, value=5)
            batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32)
        
        with col2:
            rescale_value = st.slider("Rescale Factor (1/x)", min_value=100, max_value=500, value=255)
        
        if st.button("🚀 Start Training", key="train_btn"):
            try:
                with st.spinner("Loading data..."):
                    # Data augmentation
                    datagen = ImageDataGenerator(
                        rescale=1./rescale_value,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        validation_split=0.2
                    )
                    
                    # Load training data
                    train_data = datagen.flow_from_directory(
                        dataset_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='binary',
                        subset='training'
                    )
                    
                    # Load validation data
                    val_data = datagen.flow_from_directory(
                        dataset_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='binary',
                        subset='validation'
                    )
                    
                    st.info(f"Classes: {train_data.class_indices}")
                
                with st.spinner("Building model..."):
                    # Build model
                    model = keras.Sequential()
                    
                    # First convolution layer
                    model.add(keras.layers.Conv2D(
                        filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(128, 128, 3)
                    ))
                    model.add(keras.layers.BatchNormalization())
                    
                    # First pooling layer
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    
                    # Second convolution layer
                    model.add(keras.layers.Conv2D(
                        filters=64,
                        kernel_size=(3, 3),
                        activation='relu'
                    ))
                    model.add(keras.layers.BatchNormalization())
                    
                    # Second pooling layer
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    
                    # Third convolution layer
                    model.add(keras.layers.Conv2D(
                        filters=128,
                        kernel_size=(3, 3),
                        activation='relu'
                    ))
                    model.add(keras.layers.BatchNormalization())
                    
                    # Third pooling layer
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    
                    # Fourth convolution layer
                    model.add(keras.layers.Conv2D(
                        filters=128,
                        kernel_size=(3, 3),
                        activation='relu'
                    ))
                    model.add(keras.layers.BatchNormalization())
                    
                    # Fourth pooling layer
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    
                    # Fifth convolution layer
                    model.add(keras.layers.Conv2D(
                        filters=256,
                        kernel_size=(3, 3),
                        activation='relu'
                    ))
                    model.add(keras.layers.BatchNormalization())
                    
                    # Fifth pooling layer
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                    
                    # Dropout for regularization
                    model.add(keras.layers.Dropout(0.5))
                    
                    # Flatten
                    model.add(keras.layers.Flatten())
                    
                    # Dense layers
                    model.add(keras.layers.Dense(units=256, activation='relu'))
                    model.add(keras.layers.Dropout(0.5))
                    model.add(keras.layers.Dense(units=128, activation='relu'))
                    model.add(keras.layers.Dropout(0.3))
                    
                    # Output layer
                    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
                
                # Display model summary
                st.subheader("Model Architecture")
                model_summary_text = []
                model.summary(print_fn=lambda x: model_summary_text.append(x))
                st.text("\n".join(model_summary_text))
                
                # Compile model
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Training
                st.subheader("Training Progress")
                progress_placeholder = st.empty()
                
                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_placeholder.progress(progress)
                
                history = model.fit(
                    train_data,
                    epochs=epochs,
                    validation_data=val_data,
                    verbose=0,
                    callbacks=[StreamlitCallback()]
                )
                
                # Display results
                st.success("✅ Training Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.4f}")
                with col2:
                    st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.4f}")
                
                # Plot training history
                st.subheader("Training History")
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                axes[0].plot(history.history['accuracy'], label='Training Accuracy')
                axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
                axes[0].set_title('Model Accuracy')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Accuracy')
                axes[0].legend()
                axes[0].grid(True)
                
                axes[1].plot(history.history['loss'], label='Training Loss')
                axes[1].plot(history.history['val_loss'], label='Validation Loss')
                axes[1].set_title('Model Loss')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True)
                
                st.pyplot(fig)
                
                # Evaluate on validation data
                loss, accuracy = model.evaluate(val_data, verbose=0)
                st.info(f"📊 Validation Accuracy: {accuracy:.4f}")
                
                # Save model
                model_path = "trained_model.h5"
                model.save(model_path)
                st.success(f"✅ Model saved as '{model_path}'")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                st.info("💡 Tips:\n- Ensure 'PetImages' folder has 'Cat' and 'Dog' subfolders\n- Check that images are valid (JPG/PNG)\n- Verify sufficient disk space for model saving")

elif menu == "🔮 Predict":
    st.header("Make Predictions")
    
    # Check if model exists
    model_path = "trained_model.h5"
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Model' section.")
    else:
        try:
            st.success("✅ Model loaded successfully!")
            
            # Load model
            model = keras.models.load_model(model_path)
            
            # Upload image
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                
                # Convert to RGB if necessary (handle RGBA or grayscale)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Prediction")
                    
                    try:
                        # Preprocess image
                        img_array = np.array(image.resize((128, 128))) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(img_array, verbose=0)[0][0]
                        
                        # Display results
                        if prediction > 0.5:
                            st.write("### 🐶 Dog")
                            confidence = prediction * 100
                        else:
                            st.write("### 🐱 Cat")
                            confidence = (1 - prediction) * 100
                        
                        st.metric("Confidence", f"{confidence:.2f}%")
                        
                        # Progress bar
                        st.progress(min(confidence / 100, 1.0))
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("The model file may be corrupted. Please train a new model.")

elif menu == "📈 Model Info":
    st.header("Model Information")
    
    model_path = "trained_model.h5"
    if not os.path.exists(model_path):
        st.warning("⚠️ No trained model found.")
    else:
        try:
            st.success("✅ Model found!")
            
            model = keras.models.load_model(model_path)
            
            st.subheader("Model Architecture")
            model_summary_text = []
            model.summary(print_fn=lambda x: model_summary_text.append(x))
            st.text("\n".join(model_summary_text))
            
            st.subheader("Model Details")
            st.write(f"**Total Layers**: {len(model.layers)}")
            st.write(f"**Model Input Shape**: {model.input_shape}")
            st.write(f"**Model Output Shape**: {model.output_shape}")
            
            # Calculate total parameters
            total_params = model.count_params()
            st.write(f"**Total Parameters**: {total_params:,}")
        except Exception as e:
            st.error(f"Error loading model: {e}")

st.markdown("---")
st.markdown("🚀 Developed with Streamlit | CNN - Cat vs Dog Classifier")
