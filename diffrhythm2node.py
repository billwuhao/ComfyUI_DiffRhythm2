import torch
import torchaudio
import json
import os
import re
import random
import numpy as np
import tempfile
import folder_paths
from muq import MuQMuLan
from typing import Optional
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from diffrhythm2.cfm import CFM
from diffrhythm2.backbones.dit import DiT
from diffrhythm2.bigvgan.model import Generator
from MW_utils.hf_download import download_model_with_snapshot

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS", "DiffRhythm")
cache_dir = folder_paths.get_temp_directory()

STRUCT_INFO = {
    "[start]": 500,
    "[end]": 501,
    "[intro]": 502,
    "[verse]": 503,
    "[chorus]": 504,
    "[outro]": 505,
    "[inst]": 506,
    "[solo]": 507,
    "[bridge]": 508,
    "[hook]": 509,
    "[break]": 510,
    "[stop]": 511,
    "[space]": 512
}

class CNENTokenizer():
    def __init__(self):
        curr_path = os.path.abspath(__file__)
        vocab_path = os.path.join(os.path.dirname(curr_path), "diffrhythm2/g2p/g2p/vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as file:
            self.phone2id:dict = json.load(file)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        from diffrhythm2.g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p
    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x+1 for x in token]
        return token
    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])


def cache_audio_tensor(
    cache_dir,
    audio_tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


def prepare_model(device):
    df_repo_id = "ASLP-lab/DiffRhythm2"
    df_allow_patterns = ["model.safetensors","config.json","decoder.bin","decoder.json"]
    df_local_dir = os.path.join(model_path, "DiffRhythm2")
    df_model_path = download_model_with_snapshot(repo_id=df_repo_id,local_dir=df_local_dir,allow_patterns=df_allow_patterns)

    diffrhythm2_ckpt_path = os.path.join(df_model_path, "model.safetensors")
    diffrhythm2_config_path = os.path.join(df_model_path, "config.json")

    with open(diffrhythm2_config_path, encoding="utf-8") as f:
        model_config = json.load(f)

    model_config['use_flex_attn'] = False
    diffrhythm2 = CFM(
        transformer=DiT(
            **model_config
        ),
        odeint_kwargs=dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler" # 'midpoint'
            # method="adaptive_heun"
        ),
        num_channels=model_config['mel_dim'],
        block_size=model_config['block_size'],
    )

    total_params = sum(p.numel() for p in diffrhythm2.parameters())

    diffrhythm2 = diffrhythm2.to(device)

    if diffrhythm2_ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(diffrhythm2_ckpt_path)
    else:
        ckpt = torch.load(diffrhythm2_ckpt_path, map_location='cpu')
    diffrhythm2.load_state_dict(ckpt)

    print(f"Total params: {total_params:,}")

    # prepare muq model
    try:
        from easydict import EasyDict
        main_model_dir = os.path.join(model_path, "MuQ-MuLan-large")
        local_audio_model_dir = os.path.join(model_path, "MuQ-large-msd-iter")
        local_text_model_dir = os.path.join(model_path, "xlm-roberta-base")

        main_model_path = download_model_with_snapshot(repo_id="OpenMuQ/MuQ-MuLan-large",local_dir=main_model_dir,allow_patterns=["config.json","pytorch_model.bin"])
        download_model_with_snapshot(repo_id="OpenMuQ/MuQ-large-msd-iter",
                                    local_dir=local_audio_model_dir,
                                    allow_patterns=["config.json","model.safetensors"])
        download_model_with_snapshot(repo_id="FacebookAI/xlm-roberta-base",
                                    local_dir=local_text_model_dir,
                                    allow_patterns=["config.json","sentencepiece.bpe.model","tokenizer.json","model.safetensors","tokenizer_config.json"])

        config_path = os.path.join(main_model_path, "config.json")
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = json.load(f)

        config_dict['audio_model']['name'] = local_audio_model_dir
        config_dict['text_model']['name'] = local_text_model_dir
        config_obj = EasyDict(config_dict) 

        mulan = MuQMuLan(config=config_obj, hf_hub_cache_dir=None)
        weights_path = f"{main_model_path}/pytorch_model.bin"

        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Adjust loading based on how weights are saved (e.g., remove prefixes if needed)
            mulan.load_state_dict(state_dict, strict=False) # Use strict=False initially
        except FileNotFoundError:
            raise FileNotFoundError(f"Weights file not found at {weights_path}")

    except Exception as e:
        raise
    
    mulan = mulan.to(device).eval()

    # load frontend
    lrc_tokenizer = CNENTokenizer()

    # load decoder
    decoder_ckpt_path = os.path.join(df_model_path, "decoder.bin")
    decoder_config_path = os.path.join(df_model_path, "decoder.json")

    decoder = Generator(decoder_config_path, decoder_ckpt_path)
    decoder = decoder.to(device)
    return diffrhythm2, mulan, lrc_tokenizer, decoder


STRUCT_PATTERN = re.compile(r'^\[.*?\]$')
    
def parse_lyrics(lyrics: str, lrc_tokenizer: CNENTokenizer):
    lyrics_with_time = []
    lyrics = lyrics.split("\n")
    get_start = False
    for line in lyrics:
        line = line.strip()
        if not line:
            continue
        struct_flag = STRUCT_PATTERN.match(line)
        if struct_flag:
            struct_idx = STRUCT_INFO.get(line.lower(), None)
            if struct_idx is not None:
                if struct_idx == STRUCT_INFO['[start]']:
                    get_start = True
                lyrics_with_time.append([struct_idx, STRUCT_INFO['[stop]']])
            else:
                continue
        else:
            tokens = lrc_tokenizer.encode(line.strip())
            tokens = tokens + [STRUCT_INFO['[stop]']]
            lyrics_with_time.append(tokens)
    if len(lyrics_with_time) != 0 and not get_start:
        lyrics_with_time = [[STRUCT_INFO['[start]'], STRUCT_INFO['[stop]']]] + lyrics_with_time
    return lyrics_with_time

def make_fake_stereo(audio, sampling_rate):
    left_channel = audio
    right_channel = audio.copy()
    right_channel = right_channel * 0.8
    delay_samples = int(0.01 * sampling_rate)
    right_channel = np.roll(right_channel, delay_samples)
    right_channel[:,:delay_samples] = 0
    stereo_audio = np.concatenate([left_channel, right_channel], axis=0)
    return stereo_audio


def set_all_seeds(seed):
    # import random
    # import numpy as np
    #  1. Python 内置随机模块
    random.seed(seed)
    #  2. NumPy 随机数生成器
    np.random.seed(seed)
    # 3. PyTorch CPU 和 GPU 种子
    torch.manual_seed(seed)
    # 4. 如果使用 CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        # torch.backends.cudnn.deterministic = True  # 确保卷积结果确定
        # torch.backends.cudnn.benchmark = False     # 关闭优化（牺牲速度换取确定性）


DFM = None
DECODER = None
MUQ = None
TOKENIZER = None

class DiffRhythm2Node:
    def __init__(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音乐风格提示词": ("STRING", {"default": "", "multiline": True, "tooltip": "音乐风格描述"}),
                "歌词": ("STRING", {"forceInput": True}, {"tooltip": "歌词标签有：[start]\n[end]\n[intro]\n[verse]\n[chorus]\n[outro]\n[inst]\n[solo]\n[bridge]\n[hook]\n[break]\n[stop]\n[space]"}),
                "歌曲最大长度": ("INT", {"default": 210, "min": 10, "max": 500, "step": 5}, {"tooltip": "歌曲最大长度,单位秒"}),
                },
            "optional": {
                "参考音乐": ("AUDIO", {"tooltip": "生成参考音乐风格类似歌曲"}),
                # "采样器": (["euler", "midpoint", "rk4","implicit_adams"], {"default": "euler"}),
                "步数": ("INT", {"default": 20, "min": 10, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}),
                "卸载模型": ("BOOLEAN", {"default": True}, {"tooltip": "是否在生成后卸载模型，以释放内存"}),
            },
        }

    CATEGORY = "🎤MW/MW-DiffRhythm2"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "diffrhythmgen"
    
    def diffrhythmgen(
            self,
            音乐风格提示词: str,
            歌词: str,
            歌曲最大长度: int,
            参考音乐: dict = None,
            # 采样器: str = "euler",
            步数: int = 30,
            cfg: float = 2.0,
            seed: int = 0,
            卸载模型: bool = True):

        if seed != 0:
            set_all_seeds(seed)

        global DFM, MUQ, TOKENIZER, DECODER
        if DFM is None or DECODER is None or MUQ is None or TOKENIZER is None:
            DFM, MUQ, TOKENIZER, DECODER = prepare_model(self.device)

        lyrics_token = parse_lyrics(歌词, TOKENIZER)
        lyrics_token = torch.tensor(sum(lyrics_token, []), dtype=torch.long, device=self.device)

        # preprocess style prompt
        assert 音乐风格提示词.strip() != "" or 参考音乐 is not None, "音乐风格提示词或参考音乐,请至少提供一个"

        if 参考音乐 is not None:
            prompt_wav, sr = 参考音乐["waveform"].squeeze(0), 参考音乐["sample_rate"]
            prompt_wav = torchaudio.functional.resample(prompt_wav.to(self.device), sr, 24000)
            if prompt_wav.shape[1] > 24000 * 10:
                start = random.randint(0, prompt_wav.shape[1] - 24000 * 10)
                prompt_wav = prompt_wav[:, start:start+24000*10]
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
            with torch.no_grad():
                style_prompt_embed = MUQ(wavs = prompt_wav.to(self.device))
        else:
            with torch.no_grad():
                style_prompt_embed = MUQ(texts = [音乐风格提示词]).to(self.device)

        style_prompt_embed = style_prompt_embed.to(self.device).squeeze(0)

        if self.device != 'cpu':
            DFM = DFM.half()
            DECODER = DECODER.half()
            style_prompt_embed = style_prompt_embed.half()

        with torch.inference_mode():
            latent = DFM.sample_block_cache(
                text=lyrics_token.unsqueeze(0),
                duration=int(歌曲最大长度 * 5),
                style_prompt=style_prompt_embed.unsqueeze(0),
                steps=步数,
                cfg_strength=cfg,
                process_bar=True,
            )
            latent = latent.transpose(1, 2)
            audio = DECODER.decode_audio(latent, overlap=5, chunk_size=20)
            sr = DECODER.h.sampling_rate
            audio = audio.float().cpu().numpy()
            audio_tensor = torch.from_numpy(audio).float()

        if 卸载模型:
            import gc
            DFM = None
            MUQ = None
            TOKENIZER = None
            DECODER = None
            gc.collect()
            torch.cuda.empty_cache()

        return ({"waveform": audio_tensor, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {
    "DiffRhythm2Node": DiffRhythm2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffRhythm2Node": "DiffRhythm2歌曲生成",
}