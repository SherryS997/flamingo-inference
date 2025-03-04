"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""
import os
import torch
import librosa
from copy import deepcopy
from .training.data import get_audio_features
from .training.data import int16_to_float32, float32_to_int16

from transformers import RobertaTokenizer
import wget

def create_model(
    amodel_name: str,
    tmodel_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    openai_model_cache_dir: str = os.path.expanduser("~/.cache/clip"),
    skip_params=True,
    pretrained_audio: str = "",
    pretrained_text: str = "",
    enable_fusion: bool = False,
    fusion_type: str = 'None'
    # pretrained_image: bool = False,
):
    amodel_name = amodel_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names
    pretrained_orig = pretrained
    pretrained = pretrained.lower()
    if pretrained == "openai":
        if amodel_name in _MODEL_CONFIGS:
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            raise RuntimeError(f"Model config for {amodel_name} not found.")

        # Hard Code in model name
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model = load_openai_model(
            "ViT-B-16",
            model_cfg,
            device=device,
            jit=jit,
            cache_dir=openai_model_cache_dir,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type
        )
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if amodel_name in _MODEL_CONFIGS:
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            raise RuntimeError(f"Model config for {amodel_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        # if pretrained_image:
        #     if 'timm_amodel_name' in model_cfg.get('vision_cfg', {}):
        #         # pretrained weight loading for timm models set via vision_cfg
        #         model_cfg['vision_cfg']['timm_model_pretrained'] = True
        #     else:
        #         assert False, 'pretrained image towers currently only supported for timm models'
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model_cfg["enable_fusion"] = enable_fusion
        model_cfg["fusion_type"] = fusion_type
        model = CLAP(**model_cfg)

        if pretrained:
            checkpoint_path = ""
            url = get_pretrained_url(amodel_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url, root=openai_model_cache_dir)
            elif os.path.exists(pretrained_orig):
                checkpoint_path = pretrained_orig
            if checkpoint_path:
                ckpt = load_state_dict(checkpoint_path, skip_params=True)
                model.load_state_dict(ckpt)
                param_names = [n for n, p in model.named_parameters()]
                for n in param_names:
                    print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
            else:
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
                )

        if pretrained_audio:
            if amodel_name.startswith('PANN'):
                if 'Cnn14_mAP' in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['model']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if 'spectrogram_extractor' not in key and 'logmel_extractor' not in key:
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key] = v
                elif os.path.basename(pretrained_audio).startswith('PANN'):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('finetuned'):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            elif amodel_name.startswith('HTSAT'):
                if 'HTSAT_AudioSet_Saved' in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model') and ('spectrogram_extractor' not in key
                                                            and 'logmel_extractor' not in key):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('HTSAT'):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('finetuned'):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location='cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            else:
                raise f'this audio encoder pretrained checkpoint is not support'

            model.load_state_dict(audio_ckpt, strict=False)
            logging.info(f"Loading pretrained {amodel_name} weights ({pretrained_audio}).")
            param_names = [n for n, p in model.named_parameters()]
            for n in param_names:
                print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")
            
        model.to(device=device)
        if precision == "fp16":
            assert device.type != "cpu"
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model, model_cfg

def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]

    return state_dict

class CLAP_Module(torch.nn.Module):
    def __init__(self, enable_fusion=False, device=None, amodel= 'HTSAT-tiny', tmodel='roberta') -> None:
        """Initialize CLAP Model

        Parameters
        ----------
        enable_fusion: bool
            if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false) 
        device: str
            if None, it will automatically detect the device (gpu or cpu)
        amodel: str
            audio encoder architecture, default: HTSAT-tiny
        tmodel: str
            text encoder architecture, default: roberta
        """
        super(CLAP_Module, self).__init__()
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        precision = 'fp32'

        if enable_fusion:
            fusion_type = 'aff_2d'
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type
            )
        else:
            model, model_cfg = create_model(
                amodel,
                tmodel,
                precision=precision,
                device=device,
                enable_fusion=enable_fusion
            )
        self.enable_fusion = enable_fusion
        self.model = model
        self.model_cfg = model_cfg
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result

    def load_ckpt(self, ckpt = None, model_id = -1, verbose = True):
        """Load the pretrained checkpoint of CLAP model

        Parameters
        ----------
        ckpt: str
            if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. \n 
            For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
        model_id:
            if model_id is specified, you can download our best ckpt, as:
                id = 0 --> 630k non-fusion ckpt \n
                id = 1 --> 630k+audioset non-fusion ckpt \n
                id = 2 --> 630k fusion ckpt \n
                id = 3 --> 630k+audioset fusion ckpt \n
            Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
        """
        download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
        download_names = [
            '630k-best.pt',
            '630k-audioset-best.pt',
            '630k-fusion-best.pt',
            '630k-audioset-fusion-best.pt'
        ]
        if ckpt is not None:
            print(f'Load the specified checkpoint {ckpt} from users.')
        else:
            print(f'Load our best checkpoint in the paper.')
            if model_id == -1:
                model_id = 3 if self.enable_fusion else 1
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weight_file_name = download_names[model_id]
            ckpt = os.path.join(package_dir, weight_file_name)
            if os.path.exists(ckpt):
                print(f'The checkpoint is already downloaded')
            else:
                print('Downloading laion_clap weight files...')
                ckpt = wget.download(download_link + weight_file_name, os.path.dirname(ckpt))
                print('Download completed!')
        print('Load Checkpoint...')
        ckpt = load_state_dict(ckpt, skip_params=True)
        self.model.load_state_dict(ckpt)
        if verbose:
            param_names = [n for n, p in self.model.named_parameters()]
            for n in param_names:
                print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
    
    def get_audio_embedding_from_filelist(self, x, use_tensor=False):
        """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,): 
            an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
        use_tensor: boolean:
            if True, it will return the torch tensor, preserving the gradient (default: False).
        Returns
        ----------
        audio_embed : numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        self.model.eval()
        audio_input = []
        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(f, sr=48000)           
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed


    def get_audio_embedding_from_data(self, x, use_tensor=False):
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: np.darray | torch.Tensor (N,T): 
            audio data, must be mono audio tracks.
        use_tensor: boolean:
            if True, x should be the tensor input and the output will be the tesnor, preserving the gradient (default: False).      
            Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
        Returns
        ----------
        audio embed: numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """ 
        self.model.eval()
        audio_input = []
        for audio_waveform in x:          
            # quantize
            if not use_tensor:
                audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
                audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed = self.model.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed

    def get_text_embedding(self, x, tokenizer = None, use_tensor = False):
        """get text embeddings from texts

        Parameters
        ----------
        x: List[str] (N,): 
            text list 
        tokenizer: func:
            the tokenizer function, if not provided (None), will use the default Roberta tokenizer.
        use_tensor: boolean:
            if True, the output will be the tesnor, preserving the gradient (default: False).      
        Returns
        ----------
        text_embed : numpy.darray | torch.Tensor (N,D):
            text embeddings that extracted from texts
        """ 
        self.model.eval()
        if tokenizer is not None:
            text_input = tokenizer(x)
        else:
            text_input = self.tokenizer(x)
        text_embed = self.model.get_text_embedding(text_input)
        if not use_tensor:
            text_embed = text_embed.detach().cpu().numpy()
        return text_embed
        
    