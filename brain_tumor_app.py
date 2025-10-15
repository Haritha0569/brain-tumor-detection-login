import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Global variables
model = None
IMG_SIZE = 224

def load_model_lazy():
    """Lazy load TensorFlow and model"""
    global model
    if model is None:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            model_paths = ['improved_brain_tumor_model.h5', 'brain_tumor_model.h5']
            
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        model = load_model(path)
                        print(f"✅ Successfully loaded model: {path}")
                        return model
                    except Exception as e:
                        print(f"❌ Error loading {path}: {e}")
                        continue
            
            return None
        except Exception as e:
            print(f"❌ Error importing TensorFlow: {e}")
            return None
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    if image is None:
        return None
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def create_result_overlay(image, prediction, confidence):
    """Create a visual overlay showing the prediction result"""
    try:
        # Create a copy of the original image
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Get image dimensions
        width, height = result_img.size
        
        # Set colors based on prediction
        if prediction == "🔴 Brain Tumor Detected":
            color = (255, 0, 0)  # Red
            bg_color = (255, 0, 0, 128)  # Semi-transparent red
        else:
            color = (0, 255, 0)  # Green
            bg_color = (0, 255, 0, 128)  # Semi-transparent green
        
        # Create overlay for text background
        overlay = Image.new('RGBA', result_img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Calculate text position
        margin = 20
        text_y = height - 120
        
        # Draw background rectangle for text
        overlay_draw.rectangle([
            (margin, text_y - 10), 
            (width - margin, height - margin)
        ], fill=bg_color)
        
        # Composite overlay with original image
        result_img = Image.alpha_composite(result_img.convert('RGBA'), overlay)
        result_img = result_img.convert('RGB')
        
        # Draw text on the image
        draw = ImageDraw.Draw(result_img)
        
        try:
            # Try to use a specific font
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw prediction text
        draw.text((margin + 10, text_y), prediction, fill=color, font=font_large)
        draw.text((margin + 10, text_y + 35), f"Confidence: {confidence:.1%}", fill=color, font=font_small)
        
        return result_img
    except Exception as e:
        print(f"❌ Error creating result overlay: {e}")
        return image

def predict_brain_tumor(image):
    """Make prediction on uploaded image"""
    if image is None:
        return None, "❌ Please upload an image first"
    
    try:
        # Load model if not already loaded
        current_model = load_model_lazy()
        if current_model is None:
            return None, "❌ Error: No model available. Please ensure model file exists."
        
        # Preprocess image
        processed_img = preprocess_image(image)
        if processed_img is None:
            return None, "❌ Error processing image"
        
        # Make prediction
        prediction_raw = current_model.predict(processed_img, verbose=0)[0][0]
        
        # Use optimized threshold of 0.570 based on model analysis
        threshold = 0.5
        
        if prediction_raw > threshold:
            prediction = "🔴 Brain Tumor Detected"
            confidence = prediction_raw
        else:
            prediction = "✅ No Brain Tumor Detected"
            confidence = 1 - prediction_raw
        
        # Create result overlay
        result_image = create_result_overlay(image, prediction, confidence)	
       
        # Detailed analysis text
        analysis = f"""
🔍 *AI Analysis Complete*

📊 *Prediction*: {prediction}
📈 *Confidence*: {confidence:.1%}
🧮 *Raw Score*: {prediction_raw:.3f}
⚖ *Threshold*: {threshold}

ℹ *How it works:*
- Higher scores (>{threshold}) indicate tumor presence
- Lower scores (<={threshold}) indicate normal brain tissue
- Confidence shows how certain the AI is about its prediction

⚠ *Important Disclaimer:*
This AI tool is for educational purposes only and should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical concerns.
        """
        
        return result_image, analysis.strip()
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return None, error_msg

# Enhanced mobile-responsive CSS
mobile_css = """
/* Mobile-first responsive design */
.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 10px !important;
}

/* Header styling */
h1 {
    font-size: clamp(1.5rem, 4vw, 2.5rem) !important;
    text-align: center !important;
    color: #2c3e50 !important;
    margin-bottom: 1rem !important;
}

h3 {
    font-size: clamp(1rem, 3vw, 1.5rem) !important;
    text-align: center !important;
    color: #34495e !important;
}

/* Mobile-optimized input area */
.input-image {
    max-width: 100% !important;
    margin: 0 auto !important;
}

/* Touch-friendly buttons */
button {
    min-height: 44px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
    border-radius: 8px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin: 10px 0 !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
}

/* Results area */
.output-image {
    max-width: 100% !important;
    margin: 0 auto !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

.output-text {
    background: #f8f9fa !important;
    padding: 15px !important;
    border-radius: 10px !important;
    border-left: 4px solid #007bff !important;
    margin: 10px 0 !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}

/* Public sharing indicator */
.share-status {
    background: linear-gradient(135deg, #4CAF50, #45a049) !important;
    color: white !important;
    padding: 10px !important;
    border-radius: 8px !important;
    text-align: center !important;
    margin: 10px 0 !important;
    font-weight: bold !important;
}

/* Responsive layout adjustments */
@media (max-width: 768px) {
    .gradio-container {
        padding: 5px !important;
    }
    
    .input-image, .output-image {
        max-height: 300px !important;
    }
    
    .output-text {
        font-size: 14px !important;
        padding: 10px !important;
    }
}

@media (max-width: 480px) {
    h1 {
        font-size: 1.5rem !important;
    }
    
    button {
        padding: 15px !important;
        font-size: 18px !important;
    }
    
    .input-image, .output-image {
        max-height: 250px !important;
    }
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Footer styling */
.footer {
    text-align: center !important;
    padding: 20px !important;
    color: #7f8c8d !important;
    font-size: 14px !important;
    border-top: 1px solid #ecf0f1 !important;
    margin-top: 20px !important;
}

/* Accessibility improvements */
button:focus, input:focus {
    outline: 3px solid #4CAF50 !important;
    outline-offset: 2px !important;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    button {
        background: #000 !important;
        color: #fff !important;
        border: 2px solid #fff !important;
    }
}
"""

def create_interface():
    """Create the Gradio interface with mobile optimization"""
    
    with gr.Blocks(
        css=mobile_css,
        title="🧠 AI Brain Tumor Detection - Mobile Ready",
        theme=gr.themes.Soft()
    ) as demo:
        
        # Header with public sharing status
        gr.HTML("""
        <div class="share-status">
            🌐 PUBLIC SHARING ENABLED - Universal Mobile Access
        </div>
        """)
        
        gr.Markdown("""
        # 🧠 AI Brain Tumor Detection
        ### 📱 Mobile-Optimized Medical AI Assistant
        
        Upload a brain MRI scan image and get instant AI-powered analysis. This tool uses a Convolutional Neural Network (CNN) trained to detect potential brain tumors.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Brain MRI Image")
                
                image_input = gr.Image(
                    label="Select MRI Scan",
                    type="pil",
                    sources=["upload", "webcam"],
                    height=300
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze for Brain Tumor",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 AI Analysis Results")
                
                result_image = gr.Image(
                    label="Analysis Result",
                    height=300
                )
                
                result_text = gr.Textbox(
                    label="Detailed Analysis",
                    lines=8,
                    max_lines=20
                )
        
        # Example images section
        gr.Markdown("### 📁 Try Sample Images")
        gr.Markdown("Click on any sample image below to test the AI:")
        
        example_files = []
        for file in ['tumor.jfif', 'N1.jpeg', 'Y1.jpg', 'tumor1.jpg']:
            if os.path.exists(file):
                example_files.append([file])
        
        if example_files:
            gr.Examples(
                examples=example_files,
                inputs=[image_input],
                outputs=[result_image, result_text],
                fn=predict_brain_tumor,
                cache_examples=False
            )
        
        # Information section
        gr.Markdown("""
        ### ℹ About This AI Tool
        
        This Brain Tumor Detection system uses advanced machine learning to analyze brain MRI images:
        
        - 🧠 *Technology*: Convolutional Neural Network (CNN)
        - 📊 *Training*: Trained on thousands of brain MRI images
        - 🎯 *Accuracy*: Optimized threshold for reliable detection
        - 📱 *Mobile Ready*: Works on all devices with responsive design
        - 🌐 *Shareable*: Get public links for easy access anywhere
        
        *⚠ Medical Disclaimer*: This tool is for educational and research purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical concerns.
        """)
        
        # Footer with sharing status
        gr.Markdown("""
        <div class="footer">
            🔗 Always Shared | 📱 Mobile Optimized | 🤖 AI-Powered
            <br>Brain Tumor Detection CNN - Public Access Enabled
        </div>
        """)
        
        # Set up click event
        predict_btn.click(
            fn=predict_brain_tumor,
            inputs=[image_input],
            outputs=[result_image, result_text],
            show_progress=True
        )
    
    return demo

def main():
    """Main function to launch the application"""
    
    print("🧠 Brain Tumor Detection AI - Mobile Ready Version")
    print("=" * 55)
    print("🔧 Initializing mobile-optimized interface...")
    print("🤖 Loading AI model (lazy loading enabled)...")
    print("📱 Setting up responsive design for all devices...")
    
    # Create the interface
    demo = create_interface()
    
    print("🌐 Launching application with PUBLIC SHARING enabled...")
    print("📱 Creating public link for mobile access on all devices...")
    print("🔗 You'll get a shareable link that works anywhere!")
    
    try:
        # Always launch with public sharing enabled - simplified for Windows
        demo.launch(
            share=True,             # Always enable public sharing
            inbrowser=True,
            debug=False
        )
    except Exception as e:
        print(f"❌ Error launching with public sharing: {e}")
        print("🔄 Trying local mode - you can still manually enable sharing...")
        demo.launch(
            share=False,
            inbrowser=True
        )

if _name_ == "_main_":
    main()