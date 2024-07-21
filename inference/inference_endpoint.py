import os
import yaml
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, set_seed
from src.factory import create_model_and_transforms
from data import AudioTextDataProcessor

app = Flask(__name__)

# Global variables
model = None
text_tokenizer = None
DataProcessor = None

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def prepare_tokenizer(model_config):
    tokenizer_path = model_config['tokenizer_path']
    cache_dir = model_config['cache_dir']
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if text_tokenizer.sep_token is None:
        text_tokenizer.add_special_tokens({"sep_token": "<SEP>"})
    return text_tokenizer

def prepare_model(model_config, clap_config, checkpoint_path, device_id=0):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, _ = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=False,
        gradient_checkpointing=False,
        freeze_lm_embeddings=False,
    )
    model.eval()
    model = model.to(device_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, False)

    return model

def inference(model, tokenizer, item, processed_item, inference_kwargs, device_id=0):
    filename, audio_clips, audio_embed_mask, input_ids, attention_mask = processed_item
    audio_clips = audio_clips.to(device_id, dtype=None, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=None, non_blocking=True)
    input_ids = input_ids.to(device_id, dtype=None, non_blocking=True).squeeze()

    eos_token_id = tokenizer.eos_token_id
    
    outputs = model.generate(
        audio_x=audio_clips.unsqueeze(0),
        audio_x_mask=audio_embed_mask.unsqueeze(0),
        lang_x=input_ids.unsqueeze(0),
        eos_token_id=eos_token_id,
        max_new_tokens=128,
        **inference_kwargs,
    )

    outputs_decoded = [
        tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '') for output in outputs
    ]

    return outputs_decoded[0]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    audio_file = data.get('audio_file')
    dialogue = data.get('dialogue', [])
    
    if not audio_file or not dialogue:
        return jsonify({"error": "Missing audio_file or dialogue"}), 400

    item = {
        'name': audio_file,
        'prefix': "The task is dialog.",
        'dialogue': dialogue
    }

    processed_item = DataProcessor.process(item)
    
    inference_kwargs = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    response = inference(model, text_tokenizer, item, processed_item, inference_kwargs)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    # Load configuration
    config = load_config('configs/chat.yaml')
    clap_config = config['clap_config']
    model_config = config['model_config']

    # Set up tokenizer and model
    set_seed(0)
    text_tokenizer = prepare_tokenizer(model_config)
    model = prepare_model(
        model_config=model_config, 
        clap_config=clap_config, 
        checkpoint_path="/home/sherry/Code/flamingo-inference/model_ckpts/chat.pt"
    )

    # Set up data processor
    DataProcessor = AudioTextDataProcessor(
        data_root='/home/sherry/Code/flamingo-inference/model_ckpts/datasets',
        clap_config=clap_config,
        tokenizer=text_tokenizer,
        max_tokens=512,
    )

    # Run the Flask app
    app.run(debug=True)