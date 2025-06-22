"""
BLIP-2 VQA adapter for skin lesion diagnosis.
Uses Visual Question Answering to identify skin conditions while preserving
the model's ability to recognize non-skin images and unknown conditions.
"""

import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class BLIP2VQADiagnostic(nn.Module):
    """
    BLIP-2 model adapted for skin lesion diagnosis using VQA.
    Leverages pre-trained knowledge to handle out-of-distribution images.
    """
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        use_lora: bool = True,
        lora_config: Optional[LoraConfig] = None,
        device: str = "mps"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        
        # Load BLIP-2 model and processor
        print(f"Loading {model_name}...")
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Use appropriate dtype for device
        torch_dtype = torch.float32 if device in ["mps", "cpu"] else torch.float16
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=None,  # We'll manage device placement
            low_cpu_mem_usage=True
        )
        
        # Move to device
        self.model = self.model.to(device)
        
        # Apply LoRA if requested
        if use_lora:
            if lora_config is None:
                lora_config = self.get_default_lora_config()
            
            print("Applying LoRA configuration...")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Define the skin conditions
        self.skin_conditions = [
            'melanocytic nevi',
            'melanoma',
            'benign keratosis',
            'basal cell carcinoma',
            'actinic keratosis',
            'vascular lesions',
            'dermatofibroma'
        ]
        
        # Define prompts for different stages
        self.validation_prompt = "Is this a dermatoscopic image of skin? Answer yes or no."
        self.lesion_prompt = "Does this dermatoscopic image show a skin lesion or abnormality? Answer yes, no, or unclear."
        self.classification_prompt = (
            "What type of skin lesion is shown in this dermatoscopic image? "
            f"Choose from: {', '.join(self.skin_conditions)}, or none of these."
        )
        
    @staticmethod
    def get_default_lora_config() -> LoraConfig:
        """Get default LoRA configuration optimized for BLIP-2."""
        return LoraConfig(
            r=8,  # Lower rank for smaller model
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "fc1",
                "fc2",
                "out_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.QUESTION_ANS,
        )
    
    def prepare_qa_inputs(
        self,
        images: Union[torch.Tensor, List[Image.Image]],
        questions: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for VQA training or inference.
        
        Args:
            images: Batch of images (tensors or PIL Images)
            questions: List of questions
            answers: Optional list of answers for training
        """
        # Convert tensors to PIL if needed
        if torch.is_tensor(images):
            pil_images = []
            for img_tensor in images:
                # Denormalize if needed
                if img_tensor.max() <= 1.0:
                    img_tensor = img_tensor * 255
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
        else:
            pil_images = images
        
        # Process inputs differently for training vs inference
        if answers is not None:
            # Training mode - BLIP-2 expects questions as prompts and answers as labels
            # Process questions and answers separately
            question_inputs = self.processor(
                images=pil_images,
                text=questions,  # Just the questions
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32  # Shorter for questions
            )
            
            # Process answers for labels
            answer_inputs = self.processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32  # Shorter for answers
            )
            
            # Combine inputs
            inputs = {
                'pixel_values': question_inputs['pixel_values'],
                'input_ids': question_inputs.get('input_ids'),
                'attention_mask': question_inputs.get('attention_mask'),
                'labels': answer_inputs['input_ids']
            }
        else:
            # Inference mode - questions only
            inputs = self.processor(
                images=pil_images,
                text=questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(
        self,
        images: torch.Tensor,
        questions: List[str],
        answers: Optional[List[str]] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Batch of images
            questions: Questions about the images
            answers: Ground truth answers
            return_dict: Whether to return a dictionary
        """
        if answers is not None:
            # Training mode - use a different approach for BLIP-2
            # We'll train by generating with teacher forcing
            total_loss = 0.0
            batch_size = images.shape[0]
            
            # Process each sample individually to avoid conflicts
            for i in range(batch_size):
                # Prepare single image and Q&A
                single_image = images[i:i+1]
                single_question = questions[i:i+1]
                single_answer = answers[i:i+1]
                
                # Prepare inputs with full Q&A text for teacher forcing
                full_text = [f"Question: {single_question[0]} Answer: {single_answer[0]}"]
                
                # Convert image to PIL
                if single_image.max() <= 1.0:
                    img_tensor = single_image[0] * 255
                else:
                    img_tensor = single_image[0]
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                pil_image = Image.fromarray(img_np)
                
                # Process with full text
                inputs = self.processor(
                    images=[pil_image],
                    text=full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Create labels from input_ids
                labels = inputs['input_ids'].clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                
                # Forward pass - BLIP-2 needs input_ids as well
                outputs = self.model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=labels,
                    return_dict=True
                )
                
                total_loss += outputs.loss
            
            # Average loss over batch
            return {"loss": total_loss / batch_size}
        else:
            # Inference mode
            # Prepare inputs for inference
            inputs = self.prepare_qa_inputs(images, questions, answers)
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.9,
                do_sample=False
            )
            return {"generated_ids": outputs}
    
    def generate_answer(
        self,
        image: Union[torch.Tensor, Image.Image],
        question: str,
        max_length: int = 50
    ) -> str:
        """
        Generate an answer for a single image-question pair.
        """
        # Ensure image is in correct format
        if torch.is_tensor(image) and image.dim() == 3:
            image = image.unsqueeze(0)
        
        inputs = self.prepare_qa_inputs(
            images=[image] if not torch.is_tensor(image) else image,
            questions=[question]
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                temperature=0.9,
                do_sample=False
            )
        
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return answer
    
    def diagnose(
        self,
        image: Union[torch.Tensor, Image.Image],
        return_all_stages: bool = False
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Complete diagnostic pipeline for a single image.
        
        Args:
            image: Input image
            return_all_stages: Whether to return results from all stages
        
        Returns:
            Dictionary with diagnosis results
        """
        results = {}
        
        # Stage 1: Validate if it's a dermatoscopic image
        is_derm = self.generate_answer(image, self.validation_prompt)
        results['is_dermatoscopic'] = is_derm.lower()
        
        if 'no' in is_derm.lower():
            results['diagnosis'] = 'Not a dermatoscopic image'
            results['confidence'] = 1.0
            return results
        
        # Stage 2: Check if there's a lesion
        has_lesion = self.generate_answer(image, self.lesion_prompt)
        results['has_lesion'] = has_lesion.lower()
        
        if 'no' in has_lesion.lower():
            results['diagnosis'] = 'No skin lesion detected'
            results['confidence'] = 0.9
            return results
        
        if 'unclear' in has_lesion.lower():
            results['diagnosis'] = 'Unclear - recommend professional evaluation'
            results['confidence'] = 0.5
            return results
        
        # Stage 3: Classify the lesion
        classification = self.generate_answer(image, self.classification_prompt)
        results['raw_classification'] = classification
        
        # Parse the classification
        diagnosis = None
        confidence = 0.8  # Base confidence for successful classification
        
        classification_lower = classification.lower()
        for condition in self.skin_conditions:
            if condition.lower() in classification_lower:
                diagnosis = condition
                confidence = 0.85
                break
        
        if diagnosis is None:
            if 'none' in classification_lower:
                diagnosis = 'Skin lesion present but not one of the trained categories'
                confidence = 0.7
            else:
                diagnosis = f'Uncertain classification: {classification}'
                confidence = 0.6
        
        results['diagnosis'] = diagnosis
        results['confidence'] = confidence
        
        if not return_all_stages:
            # Return only essential information
            return {
                'diagnosis': results['diagnosis'],
                'confidence': results['confidence'],
                'requires_professional_review': confidence < 0.8
            }
        
        return results
    
    def create_training_qa_pairs(
        self,
        label: str,
        augment: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Create question-answer pairs for training from a single label.
        
        Args:
            label: The ground truth label
            augment: Whether to create multiple phrasings
        
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        
        # Always include validation questions
        qa_pairs.append((self.validation_prompt, "yes"))
        qa_pairs.append((self.lesion_prompt, "yes"))
        
        # Classification with different phrasings
        qa_pairs.append((self.classification_prompt, label))
        
        if augment:
            # Alternative phrasings
            alt_questions = [
                f"What skin condition is visible in this image?",
                f"Identify the type of skin lesion shown.",
                f"This dermatoscopic image shows which condition?",
            ]
            for q in alt_questions:
                qa_pairs.append((q, label))
        
        return qa_pairs
    
    def save_pretrained(self, save_path: str):
        """Save the fine-tuned model."""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save configuration
        import json
        config = {
            'model_name': self.model_name,
            'skin_conditions': self.skin_conditions,
            'device': self.device
        }
        with open(f"{save_path}/diagnostic_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "mps"):
        """Load a fine-tuned model."""
        # Load configuration
        import json
        with open(f"{load_path}/diagnostic_config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize with saved config
        model = cls(
            model_name=config['model_name'],
            use_lora=False,  # Already has LoRA weights
            device=device
        )
        
        # Load fine-tuned weights
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(
            model.model,
            load_path,
            torch_dtype=torch.float32 if device in ["mps", "cpu"] else torch.float16
        )
        
        return model
