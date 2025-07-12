import torch
import torchaudio
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

# ====== CONFIG ======
WAV_PATH = "./models/vc/vevo/wav/output_vevotts1.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Mel Extractor from Vevo ======
from models.vc.vevo.vevo_utils import VevoInferencePipeline
from models.vc.vevo.infer_vevotts import inference_pipeline  # make sure this is initialized

# ====== Load and convert to Mel ======
waveform, sr = torchaudio.load(WAV_PATH)
mel = inference_pipeline.extract_mel_feature(waveform.to(DEVICE)).squeeze(0).cpu()

# ====== Interactive Display ======
selected = {}

def onselect(xmin, xmax):
    selected['start'] = int(xmin)
    selected['end'] = int(xmax)
    print(f"Selected mel frame region: start={selected['start']}, end={selected['end']}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
ax.set_title("Select Region to Regenerate with CFG")
ax.set_xlabel("Frame Index")
ax.set_ylabel("Mel Channel")
span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                    props=dict(alpha=0.5, facecolor='red'),
                    interactive=True)
plt.show()

# After selecting the region, you'll see output like:
# Selected mel frame region: start=540, end=600

vevo_tts(
    src_text,
    ref_wav_path,
    timbre_ref_wav_path="./models/vc/vevo/wav/mandarin_female.wav",
    output_path="./models/vc/vevo/wav/output_vevotts2.wav",
    ref_text=ref_text,
    src_language="en",
    ref_language="en",
    use_cfg=True,
    cfg_region=(720, 800),  # regenerate only this part with CFG
    cfg_scale=2.0,
)