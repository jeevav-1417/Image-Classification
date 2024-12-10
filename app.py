import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
from io import BytesIO

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("🔍 Image Classification with MobileNetV2")
    st.write("Upload an image and let the magic happen! ✨")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.markdown("""
                <div style="border: 2px solid #4CAF50; padding: 10px; text-align: center;">
                    Processing Image...
                </div>
            """, unsafe_allow_html=True)
        
        st.write("Classifying... Please wait.")
        
        with st.spinner("Loading MobileNetV2 model..."):
            model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        st.success("Classification complete!")
        
        # Display results
        results = []
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"**{label}**: {score * 100:.2f}%")
            results.append({"ID": imagenet_id, "Label": label, "Score": f"{score * 100:.2f}%"})
        
        # Allow downloading results
        results_df = pd.DataFrame(results)
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Results", data=csv, file_name="imagenet_results.csv", mime='text/csv')

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("🖼️ CIFAR-10 Image Classification")
    st.write("Upload an image to classify it into one of 10 categories.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            st.markdown("""
                <div style="border: 2px solid #FF5733; padding: 10px; text-align: center;">
                    Processing Image...
                </div>
            """, unsafe_allow_html=True)
        
        st.write("Classifying... Please wait.")
        
        with st.spinner("Loading CIFAR-10 model..."):
            model = tf.keras.models.load_model('cifar_model.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.success("Classification complete!")
        st.write(f"**Predicted Class**: {class_names[predicted_class]}")
        st.write(f"**Confidence**: {confidence * 100:.2f}%")
        
        # Allow downloading results
        results = {"Predicted Class": class_names[predicted_class], "Confidence": f"{confidence * 100:.2f}%"}
        results_df = pd.DataFrame([results])
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Results", data=csv, file_name="cifar10_results.csv", mime='text/csv')

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    st.sidebar.text_input("Enter your name", help="This will personalize your experience.")
    
    # Navigation menu
    choice = option_menu(
        "Choose Model",
        ["MobileNetV2 (ImageNet)", "CIFAR-10"],
        icons=["camera", "image"],
        menu_icon="cast",
        default_index=0
    )
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
