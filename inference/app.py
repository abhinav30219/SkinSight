"""
Gradio web interface for skin lesion diagnosis using Qwen-VL.
Provides an easy-to-use interface for uploading images and getting predictions.
"""

import sys
import os
import gradio as gr
import numpy as np
from PIL import Image
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.qwen_vl_adapter import QwenVLInference
from data.preprocessing import HAM10000Dataset


class SkinLesionDiagnosisApp:
    """Gradio application for skin lesion diagnosis."""
    
    def __init__(self, model_path: str, device: str = "mps"):
        """Initialize the application with the trained model."""
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_model()
        
        # Class information
        self.class_info = {
            'melanocytic nevi': {
                'abbr': 'nv',
                'description': 'Benign melanocytic lesions (moles)',
                'color': '#2E7D32'  # Green
            },
            'melanoma': {
                'abbr': 'mel',
                'description': 'Malignant melanoma - requires immediate medical attention',
                'color': '#D32F2F'  # Red
            },
            'benign keratosis': {
                'abbr': 'bkl',
                'description': 'Benign keratosis-like lesions (seborrheic keratosis, solar lentigo)',
                'color': '#388E3C'  # Light green
            },
            'basal cell carcinoma': {
                'abbr': 'bcc',
                'description': 'Basal cell carcinoma - most common skin cancer',
                'color': '#F57C00'  # Orange
            },
            'actinic keratosis': {
                'abbr': 'akiec',
                'description': 'Actinic keratosis and intraepithelial carcinoma (Bowen\'s disease)',
                'color': '#FFA726'  # Light orange
            },
            'vascular lesions': {
                'abbr': 'vasc',
                'description': 'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas)',
                'color': '#5C6BC0'  # Blue
            },
            'dermatofibroma': {
                'abbr': 'df',
                'description': 'Dermatofibroma - benign skin lesion',
                'color': '#8E24AA'  # Purple
            }
        }
    
    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = QwenVLInference(self.model_path, self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_probability_chart(self, probabilities: dict) -> plt.Figure:
        """Create a bar chart showing prediction probabilities."""
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Extract names and values
        names = [item[0] for item in sorted_probs]
        values = [item[1] for item in sorted_probs]
        colors = [self.class_info[name]['color'] for name in names]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1%}', va='center', fontsize=10)
        
        # Styling
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Diagnosis Probability Distribution', fontsize=14, pad=20)
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add threshold line at 50%
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        
        plt.tight_layout()
        return fig
    
    def predict(self, image):
        """Make prediction on uploaded image."""
        if image is None:
            return None, "Please upload an image", None
        
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Get prediction
            result = self.model.predict(
                image_np,
                return_probs=True,
                return_explanation=True
            )
            
            # Create probability chart
            prob_chart = self.create_probability_chart(result['probabilities'])
            
            # Format result text
            prediction_text = self.format_prediction_result(result)
            
            # Create confidence indicator
            confidence_html = self.create_confidence_indicator(result['confidence'])
            
            return prob_chart, prediction_text, confidence_html
            
        except Exception as e:
            return None, f"Error during prediction: {str(e)}", None
    
    def format_prediction_result(self, result: dict) -> str:
        """Format the prediction result as markdown text."""
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Get class info
        info = self.class_info[prediction]
        
        # Create markdown output
        output = f"""
# Diagnosis Result

## Primary Diagnosis: **{prediction.title()}**

**Confidence:** {confidence:.1%}

**Description:** {info['description']}

## All Probabilities:
"""
        
        # Add probability table
        for class_name, prob in sorted(result['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            abbr = self.class_info[class_name]['abbr']
            output += f"- **{class_name}** ({abbr}): {prob:.1%}\n"
        
        # Add explanation if available
        if 'explanation' in result and result['explanation']:
            output += f"\n## Model Explanation:\n{result['explanation']}"
        
        # Add disclaimer
        output += """

---
**⚠️ IMPORTANT DISCLAIMER:** This is an AI-powered diagnostic tool intended for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for proper diagnosis and treatment of skin conditions.
"""
        
        return output
    
    def create_confidence_indicator(self, confidence: float) -> str:
        """Create an HTML confidence indicator."""
        # Determine color based on confidence
        if confidence >= 0.8:
            color = "#4CAF50"  # Green
            level = "High"
        elif confidence >= 0.6:
            color = "#FF9800"  # Orange
            level = "Moderate"
        else:
            color = "#F44336"  # Red
            level = "Low"
        
        html = f"""
        <div style="background-color: {color}; color: white; padding: 10px; 
                    border-radius: 5px; text-align: center; font-weight: bold;">
            Confidence Level: {level} ({confidence:.1%})
        </div>
        """
        return html
    
    def create_interface(self):
        """Create the Gradio interface."""
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .gr-button {
            background-color: #2196F3;
            color: white;
        }
        .gr-button:hover {
            background-color: #1976D2;
        }
        """
        
        # Interface description
        description = """
        Upload a dermatoscopic image of a skin lesion to receive an AI-powered diagnosis.
        The model can identify 7 types of skin lesions including melanoma, basal cell carcinoma, and benign conditions.
        """
        
        # Example images info
        examples_info = """
        ### About the Model
        This diagnostic tool uses Qwen-VL 1.8B, a multimodal large language model fine-tuned on the HAM10000 dataset.
        The model has been trained to recognize:
        - Melanocytic nevi (nv)
        - Melanoma (mel)
        - Benign keratosis (bkl)
        - Basal cell carcinoma (bcc)
        - Actinic keratosis (akiec)
        - Vascular lesions (vasc)
        - Dermatofibroma (df)
        """
        
        # Create interface
        with gr.Blocks(title="Skin Lesion Diagnosis", css=custom_css) as interface:
            gr.Markdown("# 🔬 Skin Lesion Diagnosis System")
            gr.Markdown(description)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    input_image = gr.Image(
                        label="Upload Dermatoscopic Image",
                        type="pil",
                        elem_id="input-image"
                    )
                    
                    submit_btn = gr.Button(
                        "Analyze Image",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown(examples_info)
                
                with gr.Column(scale=2):
                    # Output section
                    confidence_output = gr.HTML(label="Confidence Level")
                    result_text = gr.Markdown(label="Diagnosis Details")
                    prob_chart = gr.Plot(label="Probability Distribution")
            
            # Connect the prediction function
            submit_btn.click(
                fn=self.predict,
                inputs=[input_image],
                outputs=[prob_chart, result_text, confidence_output]
            )
            
            # Add examples if available
            example_images = self.get_example_images()
            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=input_image,
                    label="Example Images"
                )
        
        return interface
    
    def get_example_images(self):
        """Get example images if available."""
        # Check if we have access to the dataset
        examples_dir = Path("../HAM10000/HAM10000_images_part_1")
        if examples_dir.exists():
            # Get a few example images
            example_files = list(examples_dir.glob("*.jpg"))[:5]
            return [str(f) for f in example_files]
        return []
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.launch(
            share=share,
            port=port,
            server_name="0.0.0.0"
        )


def main():
    """Main function to run the app."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Skin Lesion Diagnosis Web App')
    parser.add_argument(
        '--model-path',
        type=str,
        default='../training/best_model',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the app on'
    )
    
    args = parser.parse_args()
    
    # Create and launch app
    app = SkinLesionDiagnosisApp(args.model_path, args.device)
    app.launch(share=args.share, port=args.port)


if __name__ == '__main__':
    main()
