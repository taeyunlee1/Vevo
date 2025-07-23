# Copyright (c) 2023 Amphion.
# MIT License

import os
import torch
from huggingface_hub import snapshot_download
from models.vc.vevo.vevo_utils import (
    VevoInferencePipeline,
    save_audio,
)

def vevo_inpaint_ar(
    src_text,
    ref_wav_path,
    ref_text,
    blend_text,
    output_path,
    src_language="en",
    ref_language="en",
    region_tokens=(10, 20),
):
    # Generate original AR tokens and get the boundary
    ar_tokens, ref_len = inference_pipeline.run_ar(
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        style_ref_text=ref_text,
        src_lang=src_language,
        ref_lang=ref_language,
    )

    # Generate alternative AR tokens (e.g., from modified reference)
    new_tokens, _ = inference_pipeline.run_ar(
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        style_ref_text=blend_text,
        src_lang=src_language,
        ref_lang=ref_language,
    )

    # Inpaint selected region
    ar_tokens[:, ref_len + region_tokens[0]:ref_len + region_tokens[1]] =         new_tokens[:, ref_len + region_tokens[0]:ref_len + region_tokens[1]]

    # Generate audio from updated tokens
    gen_audio = inference_pipeline.run_fm_with_ar_tokens(
        ar_tokens=ar_tokens,
        timbre_ref_wav_path=ref_wav_path,
    )

    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Snapshot loading
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
    )
    ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/PhoneToVq8192")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # Initialize pipeline
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    # Inputs
    src_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall."
    ref_text = "Flip stood undecided, his ears strained to catch the slightest sound."
    blend_text = "Flip turned quickly, startled by the faint whisper behind him."

    ref_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    output_path = "./models/vc/vevo/wav/output_ar_inpaint.wav"

    vevo_inpaint_ar(
        src_text=src_text,
        ref_wav_path=ref_wav_path,
        ref_text=ref_text,
        blend_text=blend_text,
        output_path=output_path,
        region_tokens=(10, 20),  # Replace token indices 10â20 in the source portion
    )

def run_ar(self, src_text, style_ref_wav_path, style_ref_text, src_lang, ref_lang):
    """
    Generate AR tokens from concatenated [style_ref_text + src_text].
    Returns:
        - predicted_codecs: [1, T]
        - ref_token_len: int (how long the reference portion is)
    """
    style_ref_speech, _, style_ref_speech16k = load_wav(style_ref_wav_path, self.device)

    ref_ids = g2p_(style_ref_text, ref_lang)[1]
    src_ids = g2p_(src_text, src_lang)[1]

    style_ref_tensor = torch.tensor([ref_ids], dtype=torch.long).to(self.device)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
    input_ids = torch.cat([style_ref_tensor, src_tensor], dim=1)

    with torch.no_grad():
        pred_codecs = self.ar_model.generate(
            input_ids=input_ids,
            prompt_output_ids=style_ref_tensor,
            prompt_mels=self.extract_prompt_mel_feature(style_ref_speech16k),
        )

    return pred_codecs, len(ref_ids)


def run_fm_with_ar_tokens(self, ar_tokens, timbre_ref_wav_path, flow_matching_steps=32):
    """
    Run flow matching decoder using AR tokens directly.
    """
    timbre_speech, timbre_speech24k, timbre_speech16k = load_wav(timbre_ref_wav_path, self.device)
    timbre_codecs = self.extract_hubert_codec(timbre_speech16k)

    diffusion_input_codecs = torch.cat([timbre_codecs, ar_tokens], dim=1)
    cond = self.fmt_model.cond_emb(diffusion_input_codecs)
    prompt = self.extract_mel_feature(timbre_speech24k)

    with torch.no_grad():
        pred_mel = self.fmt_model.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            n_timesteps=flow_matching_steps,
        )

    mel = pred_mel.transpose(1, 2)
    audio = self.vocoder_model(mel).detach().cpu()[0]
    return audio


if blend_pred is not None:
    pred[:, start:end, :] = blend_pred[:, start:end, :]
    
    
if blend_pred is not None:
    region_len = end - start
    fade = torch.linspace(0, 1, region_len, device=pred.device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

    original = pred[:, start:end, :]
    blended = blend_pred[:, start:end, :]

    pred[:, start:end, :] = (1 - fade) * original + fade * blended

if blend_pred is not None:
    # Normalize blend_pred to match pred's std
    pred_std = pred[:, start:end, :].std()
    blend_std = blend_pred[:, start:end, :].std()
    blend_scaled = blend_pred[:, start:end, :] * (pred_std / (blend_std + 1e-6))

    pred[:, start:end, :] = (
        (1 - cfg_scale) * pred[:, start:end, :] + cfg_scale * blend_scaled
    )


def run_fm(self, predicted_codecs, timbre_ref_wav_path, bad_eps=None, region=None, cfg_scale=0.0, flow_matching_steps=32, blend_wav_path=None):
    timbre_speech, timbre_speech24k, timbre_speech16k = load_wav(timbre_ref_wav_path, self.device)
    timbre_codecs = self.extract_hubert_codec(timbre_speech16k)
    cond = self.fmt_model.cond_emb(torch.cat([timbre_codecs, predicted_codecs], dim=1))
    prompt = self.extract_mel_feature(timbre_speech24k)

    with torch.no_grad():
        pred = self.fmt_model.reverse_diffusion(
            cond=cond,
            prompt=prompt,
            n_timesteps=flow_matching_steps,
        )

        # Optional second prompt for blending
        if blend_wav_path:
            blend_speech, blend_speech24k, _ = load_wav(blend_wav_path, self.device)
            blend_prompt = self.extract_mel_feature(blend_speech24k)
            blend_pred = self.fmt_model.reverse_diffusion(
                cond=cond,
                prompt=blend_prompt,
                n_timesteps=flow_matching_steps,
            )
        else:
            blend_pred = None

        # Handle mismatch in time dimension
        if blend_pred is not None and blend_pred.shape[1] != pred.shape[1]:
            min_len = min(pred.shape[1], blend_pred.shape[1])
            print(f"[CFG Warning] Length mismatch — cropping both to {min_len} frames")
            pred = pred[:, :min_len, :]
            blend_pred = blend_pred[:, :min_len, :]

        # Region-based CFG/blending
        if cfg_scale > 0.0 and region is not None:
            start, end = region
            if end > pred.shape[1]:
                raise ValueError(f"Region end={end} exceeds available mel length={pred.shape[1]}")
            
            if blend_pred is not None:
                pred[:, start:end, :] = (
                    (1 - cfg_scale) * pred[:, start:end, :] + cfg_scale * blend_pred[:, start:end, :]
                )
            elif bad_eps is not None:
                guided = pred[:, start:end, :] + cfg_scale * (pred[:, start:end, :] - bad_eps[:, start:end, :])
                guided = torch.clamp(guided, -1.5, 1.5)
                pred[:, start:end, :] = (1 - cfg_scale) * pred[:, start:end, :] + cfg_scale * guided

    mel = pred.transpose(1, 2)
    audio = self.vocoder_model(mel).detach().cpu()[0]
    return audio


# PATCHED `vevo_utils.py`

# Add this method to VevoInferencePipeline

def cfg_forward(self, x, cond, region=None, cfg_scale=2.0):
    """Run interpolation CFG just on a region of mel"""
    full_mel = self.fmt_model(x, cond)
    if region is None:
        return full_mel  # no CFG applied

    start, end = region
    x_r = x[..., start:end]
    c_r = cond[..., start:end]
    eps_cond = self.fmt_model(x_r, c_r, use_cond=True)
    eps_uncond = self.fmt_model(x_r, c_r, use_cond=False)
    eps_cfg = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    full_mel[..., start:end] = eps_cfg
    return full_mel


# Modify signature of inference_ar_and_fm
# (add to existing list)

def inference_ar_and_fm(
    self,
    ...,
    use_cfg=False,
    cfg_region=None,
    cfg_scale=2.0,
):
    ...

    cond_emb = self.fmt_model.cond_emb(diffusion_input_codecs)
    prompt_mel = self.extract_mel_feature(timbre_ref_speech24k)
    x = self.fmt_model.sample_x(cond=cond_emb, prompt=prompt_mel, n_timesteps=flow_matching_steps)

    if use_cfg:
        predict_mel_feat = self.cfg_forward(x, cond_emb, region=cfg_region, cfg_scale=cfg_scale)
    else:
        predict_mel_feat = self.fmt_model(x, cond=cond_emb)

    synthesized_audio = (
        self.vocoder_model(predict_mel_feat.transpose(1, 2)).detach().cpu()
    )[0]
    ...
    return synthesized_audio


# PATCHED `infer_vevotts.py`

# In your call site:
vevo_tts(
    src_text,
    ref_wav_path,
    output_path="./models/vc/vevo/wav/output_fixed_segment.wav",
    ref_text=ref_text,
    src_language="en",
    ref_language="en",
    use_cfg=True,
    cfg_region=(500, 600),  # adjust this to your mel frame range
    cfg_scale=2.0,
)

# Modify vevo_tts definition:
def vevo_tts(
    src_text,
    ref_wav_path,
    timbre_ref_wav_path=None,
    output_path=None,
    ref_text=None,
    src_language="en",
    ref_language="en",
    use_cfg=False,
    cfg_region=None,
    cfg_scale=2.0,
):
    ...
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=timbre_ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
        use_cfg=use_cfg,
        cfg_region=cfg_region,
        cfg_scale=cfg_scale,
    )

    assert output_path is not None
    save_audio(gen_audio, output_path=output_path)
