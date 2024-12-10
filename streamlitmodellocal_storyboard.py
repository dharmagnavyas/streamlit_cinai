import streamlit as st
import replicate
import os
from PIL import Image, ImageDraw, ImageFont
import base64
from pathlib import Path
from io import BytesIO
import zipfile
import logging
from datetime import datetime
import json
import time
import requests
import threading
import textwrap
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FluxTrainer:
    def __init__(self, output_dir="local_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flux_version = "ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497"

    def prepare_training_data(self, uploaded_files):
        with BytesIO() as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for file in uploaded_files:
                    file_content = file.read()
                    zip_file.writestr(file.name, file_content)
                    file.seek(0)
            return f"data:application/zip;base64,{base64.b64encode(zip_buffer.getvalue()).decode()}"

    def save_model_locally(self, training, uploaded_files, params):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = self.output_dir / f"flux_model_{timestamp}"
        local_path.mkdir(parents=True, exist_ok=True)

        # Save training images
        images_path = local_path / "training_images"
        images_path.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            try:
                image = Image.open(file)
                image.save(images_path / file.name)
                file.seek(0)
            except Exception as e:
                logger.error(f"Error saving image {file.name}: {str(e)}")

        # Save configuration
        config = {
            "training_id": training.id,
            "timestamp": timestamp,
            "parameters": params,
            "base_model": self.flux_version,
            "images": [f.name for f in uploaded_files],
            "model_version": None  # Will be updated when training completes
        }
        
        with open(local_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        return local_path

    def monitor_training(self, training, config_path):
        """Monitor training progress and update config when complete"""
        while True:
            try:
                training.reload()
                if training.status == "succeeded":
                    # Update config with model version
                    config_file = Path(config_path) / "config.json"
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Get the model version from the training output
                    model_version = training.output.get('version') if training.output else None
                    config["model_version"] = model_version
                    config["status"] = "completed"
                    
                    # Save the weights URL if available
                    if training.output and training.output.get('weights'):
                        config["weights_url"] = training.output['weights']
                        # Download weights
                        weights_path = Path(config_path) / "weights.safetensors"
                        response = requests.get(config["weights_url"])
                        with open(weights_path, 'wb') as f:
                            f.write(response.content)
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    break
                elif training.status == "failed":
                    logger.error(f"Training failed: {training.error}")
                    break
                    
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error monitoring training: {str(e)}")
                time.sleep(60)  # Wait longer on error

    def generate_image(self, model_path, prompt):
        """Generate image using trained model"""
        try:
            # Load model configuration
            with open(model_path / "config.json", "r") as f:
                config = json.load(f)
            
            if not config.get("status") == "completed":
                raise Exception("Model training not completed yet")

            # Get training ID from config
            training_id = config.get("training_id")
            if not training_id:
                raise Exception("Training ID not found in config")

            # Get the training
            training = replicate.trainings.get(training_id)
            
            if training.status == "succeeded" and training.output:
                # Use the training output version for generation
                version = training.output.get('version')
                if version:
                    output = replicate.run(
                        version,
                        input={"prompt": prompt}
                    )
                    
                    if output and len(output) > 0:
                        response = requests.get(output[0])
                        return Image.open(BytesIO(response.content))
                else:
                    raise Exception("Model version not found in training output")
            else:
                raise Exception("Training not successful or output not available")
                
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise

    def add_text_to_panel(self, dialogue, image):
        """Add dialogue to panel image"""
        try:
            if not dialogue:  # If no dialogue, return original image
                return image
                
            dialogue_height = 100
            
            # Create dialogue image
            dialogue_image = self.generate_text_image(
                dialogue, 
                image.width, 
                dialogue_height,
                font_size=24,
                background_color='white'
            )
            
            # Combine images
            result_image = Image.new('RGB', (image.width, image.height + dialogue_height))
            result_image.paste(image, (0, 0))
            result_image.paste(dialogue_image, (0, image.height))
            
            return result_image
        except Exception as e:
            logger.error(f"Error in add_text_to_panel: {str(e)}")
            return image

    def generate_text_image(self, text, width, height, font_size=24, background_color='white'):
        """Generate image with text"""
        try:
            image = Image.new('RGB', (width, height), color=background_color)
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype(font="manga-temple.ttf", size=font_size)
            except IOError:
                logger.warning("Font file not found. Using default font.")
                font = ImageFont.load_default()

            wrapped_text = textwrap.wrap(text, width=45)
            y_text = 10
            
            for line in wrapped_text:
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
                    text_width = right - left
                    x = (width - text_width) // 2
                    draw.text((x, y_text), line, font=font, fill="black")
                    y_text += bottom - top + 8
                except Exception as text_error:
                    logger.error(f"Error drawing text line: {str(text_error)}")
            
            return image
        except Exception as e:
            logger.error(f"Error in generate_text_image: {str(e)}")
            return Image.new('RGB', (width, height), color=background_color)

    def create_strip(self, images):
        """Create comic strip with layout adapting to number of images"""
        padding = 20
        bottom_space_ratio = 0.3  # 30% of the height will be white space at bottom

        valid_images = [img for img in images if img is not None]
        if not valid_images:
            logger.warning("No valid images to create a strip.")
            return None

        # Determine grid size based on number of images
        num_images = len(valid_images)
        if num_images <= 4:
            columns = 2
            rows = (num_images + 1) // 2  # Will give 1 row for 1-2 images, 2 rows for 3-4 images
        else:
            columns = 2
            rows = 3  # Default to 2x3 grid for more images

        panel_width = valid_images[0].width
        panel_height = valid_images[0].height
        
        # Calculate grid dimensions
        grid_width = columns * panel_width + (columns + 1) * padding
        grid_height = rows * panel_height + (rows + 1) * padding
        
        # Add extra height for bottom white space
        total_height = int(grid_height * (1 + bottom_space_ratio))
        
        # Create white background
        result_image = Image.new("RGB", (grid_width, total_height), "white")

        # Paste images in grid
        for i, img in enumerate(valid_images):
            col = i % columns
            row = i // columns
            x = padding + col * (panel_width + padding)
            y = padding + row * (panel_height + padding)
            result_image.paste(img, (x, y))

        # Resize while maintaining aspect ratio
        target_width = 1024
        target_height = int(target_width * total_height / grid_width)
        return result_image.resize((target_width, target_height))
    
    def generate_dialogues_from_azure(self, scenario, num_frames):
        try:
            azure_key = os.getenv('azure_open_ai_key')
            endpoint_url = os.getenv('azure_endpoint_url')
            
            headers = {
                'api-key': azure_key,
                'Content-Type': 'application/json'
            }
            
            prompt = f"""Given this story scenario: "{scenario}"
            Generate EXACTLY {num_frames} dialogue lines, one for each scene. NO dialogue should be null.
            Each dialogue should describe what's visually happening in that moment while maintaining the story's atmosphere.
            Return as a JSON array of {num_frames} dialogue strings."""
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a comic dialogue writer who creates contextually accurate dialogues matching scene actions"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            }
            
            response = requests.post(endpoint_url, headers=headers, json=payload)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            try:
                dialogues = json.loads(content)
                while len(dialogues) < num_frames:
                    dialogues.append(None)
                return dialogues[:num_frames]
            except json.JSONDecodeError:
                logger.error("Failed to parse Azure OpenAI response as JSON")
                return [None] * num_frames
                
        except Exception as e:
            logger.error(f"Error generating dialogues from Azure: {str(e)}")
            return [None] * num_frames

    def generate_story_panels(self, model_path, scenario, num_frames):
        """Generate story panels using the selected model"""
        try:
            panels = []
            sentences = scenario.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Generate dialogues using Azure OpenAI
            dialogues = self.generate_dialogues_from_azure(scenario, num_frames)
            for i in range(num_frames):
                idx = min(i, len(sentences) - 1)
                scene_description = sentences[idx]
                dialogue = dialogues[i] if dialogues and i < len(dialogues) else None
                
                # Generate image using selected model
                image = self.generate_image(model_path, scene_description)
                
                if image:
                    # Add dialogue to image
                    image_with_text = self.add_text_to_panel(dialogue, image)
                    
                    panels.append({
                        'image': image_with_text,
                        'description': scene_description,
                        'dialogue': dialogue,
                        'number': i + 1
                    })
            
            return panels
            
        except Exception as e:
            logger.error(f"Error generating story panels: {str(e)}")
            raise
    

    def train_model(self, username, model_name, uploaded_files, params):
        try:
            destination = f"{username}/{model_name}"
            data_uri = self.prepare_training_data(uploaded_files)
            
            training = replicate.trainings.create(
                destination=destination,
                version=self.flux_version,
                input={
                    "steps": params["steps"],
                    "lora_rank": params["lora_rank"],
                    "optimizer": "adamw8bit",
                    "batch_size": params["batch_size"],
                    "resolution": params["resolution"],
                    "autocaption": True,
                    "input_images": data_uri,
                    "trigger_word": params["trigger_word"],
                    "learning_rate": params["learning_rate"],
                    "caption_dropout_rate": 0.05
                }
            )
            
            # Save locally
            local_path = self.save_model_locally(training, uploaded_files, params)
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_training,
                args=(training, local_path)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return training, local_path
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def check_training_status(model_path):
    """Check if model training is complete"""
    try:
        config_file = Path(model_path) / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get("status") == "completed"
    except Exception as e:
        logger.error(f"Error checking training status: {str(e)}")
    return False

def main():
    st.title("Flux Model Training and Testing Interface")
    
    # Initialize session state
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = None
    if 'show_test_interface' not in st.session_state:
        st.session_state.show_test_interface = False
    
    # Sidebar for API configuration
    with st.sidebar:
        api_token = st.text_input("Enter Replicate API Token", type="password")
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token
            
        username = st.text_input("Your Replicate Username")
        model_name = st.text_input("Model Name (e.g., my-lora-model)")
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Train Model", "Test Model"])
    
    with tab1:
        st.header("Train New Model")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            params = {
                "trigger_word": st.text_input("Trigger Word", value="TOK"),
                "steps": st.slider("Training Steps", min_value=500, max_value=4000, value=1000),
                "lora_rank": st.slider("LoRA Rank", min_value=1, max_value=128, value=16),
                "learning_rate": st.number_input("Learning Rate", value=0.0004, format="%.4f"),
            }
        
        with col2:
            params.update({
                "resolution": st.selectbox("Resolution", ["512", "768", "1024"]),
                "batch_size": st.number_input("Batch Size", value=1, min_value=1),
            })
        
        uploaded_files = st.file_uploader(
            "Upload training images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        trainer = FluxTrainer()
        
        if st.button("Start Training"):
            if not api_token:
                st.error("Please enter your Replicate API Token")
                return
            if not username or not model_name:
                st.error("Please enter both your Replicate username and model name")
                return
            if not uploaded_files:
                st.error("Please upload training images")
                return
                
            try:
                training, local_path = trainer.train_model(
                    username,
                    model_name,
                    uploaded_files,
                    params
                )
                
                st.session_state.current_model_path = str(local_path)
                
                st.success(f"""
                Training started successfully!
                Your model and training data have been saved locally at:
                {local_path}
                Training ID: {training.id}
                """)
                
                # Create placeholder for status updates
                status_placeholder = st.empty()
                
                # Check training status periodically
                while True:
                    if check_training_status(local_path):
                        status_placeholder.success("Training completed! You can now test your model.")
                        st.session_state.show_test_interface = True
                        st.rerun()
                        break
                    status_placeholder.info("Training in progress... Please wait.")
                    time.sleep(30)
                    
            except Exception as e:
                st.error(f"Error during training process: {str(e)}")
                logger.error(f"Training error: {str(e)}")
    
    with tab2:
        st.header("Test Trained Model")
        
        # List available models
        model_dirs = list(Path("local_models").glob("flux_model_*"))
        if not model_dirs:
            st.warning("No trained models found. Please train a model first.")
        else:
            selected_model = st.selectbox(
                "Select model",
                model_dirs,
                format_func=lambda x: x.name
            )
            
            # Add radio button for generation type
            generation_type = st.radio(
                "Generation type",
                ["Single Image", "Storyboard"],
                horizontal=True
            )
            
            if generation_type == "Single Image":
                prompt = st.text_area("Enter your prompt")
                
                if st.button("Generate Image"):
                    if not api_token:
                        st.error("Please enter your Replicate API Token")
                        return
                        
                    try:
                        if not check_training_status(selected_model):
                            st.warning("Model training not completed yet. Please wait.")
                            return
                            
                        trainer = FluxTrainer()
                        with st.spinner("Generating image..."):
                            image = trainer.generate_image(selected_model, prompt)
                            if image:
                                st.image(image, caption="Generated Image")
                            else:
                                st.error("Failed to generate image")
                                
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
            
            else:  # Storyboard mode
                scenario = st.text_area(
                    "Enter your story scenario",
                    help="Describe your story scenario. Each sentence will be converted into a panel. Dialogues will be automatically generated."
                )
                
                num_frames = st.slider(
                    "Number of frames",
                    min_value=2,
                    max_value=8,
                    value=4
                )
                
                if st.button("Generate Storyboard"):
                    if not api_token:
                        st.error("Please enter your Replicate API Token")
                        return
                        
                    if not scenario:
                        st.error("Please enter a story scenario")
                        return
                        
                    try:
                        if not check_training_status(selected_model):
                            st.warning("Model training not completed yet. Please wait.")
                            return
                            
                        trainer = FluxTrainer()
                        with st.spinner("Generating storyboard with AI-generated dialogues..."):
                            panels = trainer.generate_story_panels(
                                selected_model,
                                scenario,
                                num_frames
                            )
                            
                            # Display panels
                            st.subheader("Generated Storyboard")
                            cols = st.columns(min(2, num_frames))
                            for idx, panel in enumerate(panels):
                                col_idx = idx % len(cols)
                                with cols[col_idx]:
                                    st.image(panel['image'], caption=f"Panel {panel['number']}")
                                    # Only show dialogue if it exists, without any other text
                                    if panel.get('dialogue'):
                                        st.markdown(panel['dialogue'])
                            
                            # Create and save strip
                            strip_image = trainer.create_strip([p['image'] for p in panels])
                            
                            if strip_image:
                                st.subheader("Complete Comic Strip")
                                st.image(strip_image, caption="Final Comic Strip")
                            
                            # Save everything
                            save_dir = Path("generated_storyboards")
                            save_dir.mkdir(exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            story_dir = save_dir / f"storyboard_{timestamp}"
                            story_dir.mkdir()
                            
                            # Save individual panels and strip
                            for panel in panels:
                                panel['image'].save(story_dir / f"panel_{panel['number']}.png")
                            if strip_image:
                                strip_image.save(story_dir / "comic_strip.png")
                            
                            # Save metadata
                            metadata = {
                                'scenario': scenario,
                                'num_frames': num_frames,
                                'timestamp': timestamp,
                                'model_used': str(selected_model),
                                'panels': [{
                                    'number': p['number'],
                                    'description': p['description'],
                                    'dialogue': p.get('dialogue')
                                } for p in panels]
                            }
                            
                            with open(story_dir / "metadata.json", "w") as f:
                                json.dump(metadata, f, indent=2)
                            
                            # Create ZIP file
                            zip_path = story_dir / f"storyboard_{timestamp}.zip"
                            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                                # Add panel images
                                for i in range(num_frames):
                                    panel_path = story_dir / f"panel_{i+1}.png"
                                    zip_file.write(panel_path, panel_path.name)
                                # Add strip
                                if strip_image:
                                    zip_file.write(story_dir / "comic_strip.png", "comic_strip.png")
                                # Add metadata
                                zip_file.write(story_dir / "metadata.json", "metadata.json")
                            
                            # Offer download options
                            st.success(f"Storyboard generated and saved to {story_dir}")
                            
                            with open(zip_path, "rb") as f:
                                zip_data = f.read()
                            
                            st.download_button(
                                label="Download Complete Storyboard Package",
                                data=zip_data,
                                file_name=f"storyboard_{timestamp}.zip",
                                mime="application/zip"
                            )
                            
                            # Display summary
                            st.subheader("Storyboard Summary")
                            st.json(metadata)
                            
                    except Exception as e:
                        st.error(f"Error generating storyboard: {str(e)}")
                        logger.error(f"Storyboard generation error: {str(e)}")

if __name__ == "__main__":
    main()