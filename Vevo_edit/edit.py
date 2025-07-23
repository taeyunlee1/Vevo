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
