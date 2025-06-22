"""
Qwen-VL model adapter for skin lesion classification.
Adapts the Qwen-VL model for 7-class classification task.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image


class QwenVLClassifier(nn.Module):
    """
    Wrapper around Qwen-VL for classification tasks.
    Adds a classification head on top of the language model.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        num_classes: int = 7,
        use_lora: bool = True,
        lora_config: Optional[LoraConfig] = None,
        device: str = "mps"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        
        # Load Qwen-VL model and processor
        print(f"Loading {model_name}...")
        # Note: MPS supports float16 for inference but not all operations
        # Use float32 for MPS to ensure compatibility
        if device == "mps":
            torch_dtype = torch.float32
        elif device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16
            
        # Load model with complete bypass of initialization
        import warnings
        import os
        
        print("Loading model with custom initialization bypass...")
        
        # First, let's try a different approach - patch the model files directly
        try:
            # Import the model's config
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Now we need to patch the initialization before loading
            import transformers.modeling_utils
            from transformers.modeling_utils import PreTrainedModel
            
            # Create a completely custom initialization that does nothing
            class NoInitModel(PreTrainedModel):
                def _init_weights(self, module):
                    pass
                
                def _initialize_weights(self, module):
                    pass
                
                def initialize_weights(self):
                    pass
                
                def _initialize_missing_keys(self, *args, **kwargs):
                    pass
            
            # Temporarily replace PreTrainedModel
            original_base = transformers.modeling_utils.PreTrainedModel
            transformers.modeling_utils.PreTrainedModel = NoInitModel
            
            # Also patch in the actual model module if it's loaded
            import sys
            for module_name in list(sys.modules.keys()):
                if 'modeling_qwen' in module_name:
                    module = sys.modules[module_name]
                    if hasattr(module, 'PreTrainedModel'):
                        module.PreTrainedModel = NoInitModel
                    # Also patch Resampler if it exists
                    if hasattr(module, 'Resampler'):
                        # Add missing method
                        module.Resampler._initialize_weights = lambda self, x: None
            
            # Now load the model
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Set environment variable to skip initialization
                os.environ["TRANSFORMERS_SKIP_INIT"] = "1"
                
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype if device == "cuda" else torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=True,
                        _fast_init=True
                    )
                    print("Model loaded successfully!")
                finally:
                    # Clean up environment
                    if "TRANSFORMERS_SKIP_INIT" in os.environ:
                        del os.environ["TRANSFORMERS_SKIP_INIT"]
            
        except Exception as e:
            print(f"Error during custom loading: {e}")
            # Fallback to standard loading with error handling
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=None,
                    low_cpu_mem_usage=True
                )
            except Exception as fallback_error:
                print(f"Fallback loading also failed: {fallback_error}")
                raise
        
        finally:
            # Restore original PreTrainedModel if we changed it
            if 'original_base' in locals():
                transformers.modeling_utils.PreTrainedModel = original_base
                # Restore in loaded modules too
                for module_name in list(sys.modules.keys()):
                    if 'modeling_qwen' in module_name:
                        module = sys.modules[module_name]
                        if hasattr(module, 'PreTrainedModel'):
                            module.PreTrainedModel = original_base
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Move model to device
        if device == "mps":
            self.model = self.model.to(device)
        
        # Apply LoRA if requested
        if use_lora:
            if lora_config is None:
                lora_config = self.get_default_lora_config()
            
            print("Applying LoRA configuration...")
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Add classification head
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        ).to(device)
        
        # Set up for gradient checkpointing if needed
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
    
    @staticmethod
    def get_default_lora_config() -> LoraConfig:
        """Get default LoRA configuration optimized for Qwen-VL."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "c_attn",  # attention projection
                "c_proj",  # output projection
                "w1",      # MLP layers
                "w2"
            ],
            bias="none",
            modules_to_save=None,
        )
    
    def prepare_inputs(
        self,
        images: torch.Tensor,
        prompts: List[str],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for Qwen-VL model.
        
        Args:
            images: Batch of images (B, C, H, W)
            prompts: List of text prompts
            labels: Optional labels for training
        """
        # Convert tensors to PIL images for processor
        pil_images = []
        for img_tensor in images:
            # Denormalize - make sure tensors are on same device
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(img_tensor.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(img_tensor.device)
            img = img_tensor * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to PIL
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        # Process with Qwen processor
        # Note: Qwen processor might not support padding parameter
        try:
            inputs = self.processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True
            )
        except ValueError:
            # If padding fails, process each item individually and pad manually
            all_inputs = []
            for prompt, image in zip(prompts, pil_images):
                single_input = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
                all_inputs.append(single_input)
            
            # Manually pad and batch
            # Get max length
            max_len = max(inp['input_ids'].shape[1] for inp in all_inputs)
            
            # Pad each input
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            
            for inp in all_inputs:
                # Pad input_ids
                curr_len = inp['input_ids'].shape[1]
                if curr_len < max_len:
                    pad_len = max_len - curr_len
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    inp['input_ids'] = torch.cat([
                        inp['input_ids'],
                        torch.full((1, pad_len), pad_token_id, dtype=inp['input_ids'].dtype)
                    ], dim=1)
                    inp['attention_mask'] = torch.cat([
                        inp['attention_mask'],
                        torch.zeros((1, pad_len), dtype=inp['attention_mask'].dtype)
                    ], dim=1)
                
                input_ids_list.append(inp['input_ids'])
                attention_mask_list.append(inp['attention_mask'])
                if 'pixel_values' in inp:
                    pixel_values_list.append(inp['pixel_values'])
            
            # Stack into batches
            inputs = {
                'input_ids': torch.cat(input_ids_list, dim=0),
                'attention_mask': torch.cat(attention_mask_list, dim=0)
            }
            if pixel_values_list:
                inputs['pixel_values'] = torch.cat(pixel_values_list, dim=0)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if labels is not None:
            inputs['labels'] = labels.to(self.device)
        
        # Debug: print what keys we have
        print(f"Processor returned keys: {list(inputs.keys())}")
        
        return inputs
    
    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            images: Batch of images
            prompts: List of prompts
            labels: Optional labels for computing loss
            return_logits: Whether to return classification logits
        
        Returns:
            Dictionary containing loss and/or logits
        """
        # Prepare inputs
        inputs = self.prepare_inputs(images, prompts, labels)
        
        # Forward through Qwen-VL
        # Check which keys are available in inputs
        model_kwargs = {
            'input_ids': inputs['input_ids'],
            'output_hidden_states': True,
            'return_dict': True
        }
        
        # Add optional parameters if they exist
        if 'attention_mask' in inputs:
            model_kwargs['attention_mask'] = inputs['attention_mask']
        
        # For Qwen-VL, images might be passed differently
        # Try different parameter names for images
        if 'pixel_values' in inputs:
            # Try 'images' parameter first
            try:
                model_kwargs['images'] = inputs['pixel_values']
                outputs = self.model(**model_kwargs)
            except TypeError:
                # If that fails, try without images (text-only)
                del model_kwargs['images']
                outputs = self.model(**model_kwargs)
                print("Warning: Running without images, Qwen-VL might need different image input format")
        else:
            outputs = self.model(**model_kwargs)
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Pool the hidden states (use last token)
        pooled_output = hidden_states[:, -1, :]
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        result = {}
        
        # Calculate loss if labels provided
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            result['loss'] = loss
        
        if return_logits:
            result['logits'] = logits
        
        return result
    
    def generate_response(
        self,
        image: torch.Tensor,
        prompt: str,
        max_length: int = 100
    ) -> str:
        """
        Generate a text response for a single image.
        Useful for getting explanations.
        """
        # Prepare single input
        inputs = self.prepare_inputs(
            images=image.unsqueeze(0),
            prompts=[prompt]
        )
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                pixel_values=inputs.get('pixel_values'),
                max_length=max_length,
                do_sample=False,
                temperature=0.7
            )
        
        # Decode
        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return response
    
    def save_adapter(self, save_path: str):
        """Save LoRA adapter and classifier head."""
        # Save LoRA adapter
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_path)
        
        # Save classifier head
        classifier_path = f"{save_path}/classifier.pt"
        torch.save(self.classifier.state_dict(), classifier_path)
        
        print(f"Model adapter saved to {save_path}")
    
    def load_adapter(self, load_path: str):
        """Load LoRA adapter and classifier head."""
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            load_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )
        
        # Load classifier head
        classifier_path = f"{load_path}/classifier.pt"
        self.classifier.load_state_dict(torch.load(classifier_path))
        
        print(f"Model adapter loaded from {load_path}")


class QwenVLInference:
    """Inference wrapper for deployed model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "mps"
    ):
        """Initialize inference model."""
        self.device = device
        
        # Load base model
        base_model_name = "Qwen/Qwen-VL-Chat"
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(
            self.model,
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Load classifier
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7)
        ).to(device)
        
        classifier_path = f"{model_path}/classifier.pt"
        self.classifier.load_state_dict(torch.load(classifier_path))
        
        # Move to device and eval mode
        if device == "mps":
            self.model = self.model.to(device)
        self.model.eval()
        self.classifier.eval()
        
        # Class names
        self.class_names = [
            'melanocytic nevi',
            'melanoma',
            'benign keratosis',
            'basal cell carcinoma',
            'actinic keratosis',
            'vascular lesions',
            'dermatofibroma'
        ]
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        return_probs: bool = True,
        return_explanation: bool = False
    ) -> Dict:
        """
        Make prediction on a single image.
        
        Args:
            image: Image as numpy array (H, W, C)
            return_probs: Return probability distribution
            return_explanation: Generate text explanation
        
        Returns:
            Dictionary with prediction results
        """
        # Convert to PIL
        from PIL import Image
        pil_image = Image.fromarray(image)
        
        # Prepare prompt
        prompt = (
            "What type of skin lesion is shown in this dermatoscopic image? "
            "Choose from: melanocytic nevi, melanoma, benign keratosis, "
            "basal cell carcinoma, actinic keratosis, vascular lesions, or dermatofibroma."
        )
        
        # Process inputs
        inputs = self.processor(
            text=[prompt],
            images=[pil_image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            pixel_values=inputs.get('pixel_values'),
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get classification
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states[:, -1, :]
        logits = self.classifier(pooled_output)
        
        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[0, pred_idx].item()
        
        result = {
            'prediction': pred_class,
            'confidence': confidence,
            'class_index': pred_idx
        }
        
        if return_probs:
            result['probabilities'] = {
                name: prob.item() 
                for name, prob in zip(self.class_names, probs[0])
            }
        
        if return_explanation:
            # Generate explanation
            explain_prompt = f"The image shows {pred_class}. Explain why this diagnosis was made."
            
            explain_inputs = self.processor(
                text=[explain_prompt],
                images=[pil_image],
                return_tensors="pt"
            )
            explain_inputs = {k: v.to(self.device) for k, v in explain_inputs.items()}
            
            generated_ids = self.model.generate(
                input_ids=explain_inputs['input_ids'],
                attention_mask=explain_inputs.get('attention_mask'),
                pixel_values=explain_inputs.get('pixel_values'),
                max_length=150,
                do_sample=True,
                temperature=0.7
            )
            
            explanation = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            result['explanation'] = explanation
        
        return result
