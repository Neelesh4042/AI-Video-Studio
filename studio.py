
# ===== Neelesh AI Video Studio v2.4 (PixVerse jaisa, personal use) =====
# Text->Video, Photo->Video, Video Extend, Hindi prompt support
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re, time, glob, gc, torch, gradio as gr
from PIL import Image, ImageOps
from diffusers import AutoencoderKLWan, WanPipeline, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video
from transformers import AutoTokenizer, UMT5EncoderModel

T2V_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VACE_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
OUT_DIR = "/kaggle/working/videos"
os.makedirs(OUT_DIR, exist_ok=True)

# purana model memory se hatao (cell dobara chalane par OOM se bachne ke liye)
for _n in ["text_encoder", "pipe", "vpipe", "vace_tr", "vae"]:
    globals().pop(_n, None)
gc.collect(); torch.cuda.empty_cache()

NGPU = torch.cuda.device_count()
print("GPUs:", NGPU)
assert NGPU >= 1, "GPU nahi mila! Settings > Accelerator > GPU T4 x2 chuniye"
DTYPE = torch.float16
GEN_DEV = "cuda:0"
TE_DEV = "cuda:1" if NGPU > 1 else "cuda:0"

print("Model load ho raha hai (pehli baar VACE download ~5 min)...")
tokenizer = AutoTokenizer.from_pretrained(T2V_ID, subfolder="tokenizer")
text_encoder = UMT5EncoderModel.from_pretrained(
    T2V_ID, subfolder="text_encoder", torch_dtype=DTYPE,
    low_cpu_mem_usage=True, device_map={"": TE_DEV}).eval()
gc.collect()
print("Text encoder ready")

vae = AutoencoderKLWan.from_pretrained(T2V_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(T2V_ID, vae=vae, text_encoder=None, tokenizer=None,
                                   torch_dtype=DTYPE, low_cpu_mem_usage=True)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
pipe.to(GEN_DEV)
for _f in ("enable_tiling", "enable_slicing"):   # VAE ki memory kam (OOM fix)
    try: getattr(pipe.vae, _f)()
    except Exception as e: print(_f, "skip:", e)
print("Text->Video ready")

vace_tr = WanVACETransformer3DModel.from_pretrained(VACE_ID, subfolder="transformer",
                                                    torch_dtype=DTYPE, low_cpu_mem_usage=True)
vpipe = WanVACEPipeline(tokenizer=None, text_encoder=None, vae=pipe.vae,
                        scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0),
                        transformer=vace_tr)
vace_tr.to("cpu")   # ek samay par GPU0 par sirf ek model (OOM fix)
gc.collect(); torch.cuda.empty_cache()
ACTIVE = "t2v"

def use(which):
    """GPU0 par zaroorat wala model rakho, doosra CPU par"""
    global ACTIVE
    if ACTIVE == which:
        return
    off, on = (vace_tr, pipe.transformer) if which == "t2v" else (pipe.transformer, vace_tr)
    off.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    on.to(GEN_DEV); ACTIVE = which
    if 'sync_turbo' in globals(): sync_turbo()
# ---------- Turbo: CausVid LoRA (6 steps, bina CFG = ~5x tez), khud merge karte hain ----------
TURBO, WANT_TURBO, MERGED, LORA = False, False, {}, []
def _map(k):
    k = k.replace("diffusion_model.", "")
    for a_, b_ in [("self_attn.o", "attn1.to_out.0"), ("cross_attn.o", "attn2.to_out.0"),
                   ("self_attn.", "attn1."), ("cross_attn.", "attn2."),
                   ("ffn.0", "ffn.net.0.proj"), ("ffn.2", "ffn.net.2"), ("norm3", "norm2"),
                   ("head.head", "proj_out"),
                   ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
                   ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
                   ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
                   ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
                   ("time_projection.1", "condition_embedder.time_proj")]:
        k = k.replace(a_, b_)
    for qq in ("q", "k", "v"):
        k = k.replace(f"attn1.{qq}.", f"attn1.to_{qq}.").replace(f"attn2.{qq}.", f"attn2.to_{qq}.")
    return k
try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    _sd = load_file(hf_hub_download("Kijai/WanVideo_comfy", "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors"))
    for k in _sd:
        if k.endswith(".lora_down.weight"):
            base = k[:-len(".lora_down.weight")]
            LORA.append((_map(base) + ".weight", "lora", _sd[base + ".lora_up.weight"].float(), _sd[k].float()))
        elif k.endswith(".diff_b"):
            LORA.append((_map(k[:-7]) + ".bias", "add", _sd[k].float(), None))
        elif k.endswith(".diff"):
            LORA.append((_map(k[:-5]) + ".weight", "add", _sd[k].float(), None))
    _params = dict(pipe.transformer.named_parameters())
    _miss = [n for n, *_ in LORA if n not in _params]
    print(f"Turbo LoRA: {len(LORA)} parts, {len(_miss)} match nahi hue", _miss[:5])
    LORA = [x for x in LORA if x[0] not in _miss]
    TURBO = len(LORA) > 100
    print("Turbo ready" if TURBO else "Turbo off")
except Exception as e:
    print("Turbo LoRA load nahi hua (normal mode chalega):", repr(e))

@torch.no_grad()
def _merge(model, sign):
    params = dict(model.named_parameters())
    for name, kind, up, down in LORA:
        w = params.get(name)
        if w is None:
            continue
        d = (up.to(w.device) @ down.to(w.device)) if kind == "lora" else up.to(w.device)
        w.add_((sign * d).to(w.dtype))

def sync_turbo():
    m = pipe.transformer if ACTIVE == "t2v" else vace_tr
    on = bool(MERGED.get(id(m)))
    if TURBO and WANT_TURBO != on:
        _merge(m, 1.0 if WANT_TURBO else -1.0); MERGED[id(m)] = WANT_TURBO

def set_turbo(on):
    global WANT_TURBO
    WANT_TURBO = on; sync_turbo()
print("Photo->Video + Extend ready")

# ---------- Hindi -> English ----------
def to_english(text):
    text = (text or "").strip()
    if re.search(r"[\u0900-\u097F]", text):
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source="auto", target="en").translate(text), True
        except Exception as e:
            print("Translate error:", e)
    return text, False

@torch.no_grad()
def encode(text, max_len=512):
    t = tokenizer([text], padding="max_length", max_length=max_len, truncation=True,
                  add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    ids, mask = t.input_ids.to(TE_DEV), t.attention_mask.to(TE_DEV)
    emb = text_encoder(ids, mask).last_hidden_state.float()
    n = int(mask.sum())
    out = torch.zeros(1, max_len, emb.shape[-1])
    out[:, :n] = emb[:, :n].cpu()
    return torch.nan_to_num(out).to(GEN_DEV, DTYPE)

print("Model ready!")

NEG = ("Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, "
       "JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, "
       "messy background, three legs, walking backwards, still picture")
STYLES = {
    "None": "",
    "Cinematic": ", cinematic lighting, film grain, shallow depth of field, epic shot",
    "Realistic": ", photorealistic, natural light, highly detailed, 4k",
    "Anime": ", anime style, vibrant colors, studio quality animation",
    "3D Cartoon": ", 3D cartoon style, pixar like, soft lighting, cute",
    "Cyberpunk": ", cyberpunk, neon lights, night city, rain reflections",
    "Mythic Dark": ", dark mythological Indian fantasy, dramatic lightning, smoke, cinematic, realistic",
}
RATIOS = {"16:9": (480, 832), "9:16": (832, 480), "1:1": (624, 624)}
LENGTHS = {"2 sec (fast)": 33, "3 sec": 49, "5 sec (slow)": 81}
QUALITY = {"Turbo (6 steps, tez)": 6, "Fast (20 steps)": 20, "Better (30 steps)": 30, "Best (50 steps)": 50}
DEFAULT_Q = "Turbo (6 steps, tez)"

def qset(quality):
    """steps, guidance; Turbo me LoRA on aur CFG off"""
    turbo = quality.startswith("Turbo") and TURBO
    set_turbo(turbo)
    if quality.startswith("Turbo") and not TURBO:
        return 20, 5.0
    return QUALITY[quality], (1.0 if turbo else 5.0)
GRAY = (128, 128, 128)
KEEP = 5  # extend ke liye purane video ke aakhri frames

def history():
    return sorted(glob.glob(f"{OUT_DIR}/*.mp4"), reverse=True)

def new_path(tag, seed):
    return f"{OUT_DIR}/{time.strftime('%Y%m%d_%H%M%S')}_{tag}_{seed}.mp4"

def get_seed(seed):
    seed = int(seed) if seed is not None and int(seed) >= 0 else int(time.time()) % 2**31
    return seed, torch.Generator("cpu").manual_seed(seed)

def fit(img, h, w):
    return ImageOps.fit(img.convert("RGB"), (w, h), Image.LANCZOS)

def auto_ratio(img):
    r = img.width / img.height
    return "16:9" if r > 1.25 else ("9:16" if r < 0.8 else "1:1")

def prep_prompt(prompt, style):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt likhiye")
    en, tr = to_english(prompt)
    return en + STYLES[style], (f"\n\nEnglish prompt: _{en}_" if tr else "")

def done(path, t0, seed, note):
    gc.collect(); torch.cuda.empty_cache()
    return path, f"Ban gaya! {int(time.time()-t0)} sec | seed {seed}{note}", history()

# ---------- 1) Text -> Video ----------
def gen_text(prompt, style, ratio, length, quality, seed, progress=gr.Progress(track_tqdm=True)):
    full, note = prep_prompt(prompt, style)
    h, w = RATIOS[ratio]; seed, g = get_seed(seed); t0 = time.time()
    use("t2v"); st, cf = qset(quality)
    frames = pipe(prompt_embeds=encode(full), negative_prompt_embeds=encode(NEG),
                  height=h, width=w, num_frames=LENGTHS[length],
                  num_inference_steps=st, guidance_scale=cf, generator=g).frames[0]
    path = new_path("t2v", seed); export_to_video(frames, path, fps=16)
    return done(path, t0, seed, note)

# ---------- 2) Photo -> Video (optional last photo) ----------
def gen_image(img, last_img, prompt, style, length, quality, seed, progress=gr.Progress(track_tqdm=True)):
    if img is None:
        raise gr.Error("Pehle photo upload kijiye")
    full, note = prep_prompt(prompt, style)
    h, w = RATIOS[auto_ratio(img)]; n = LENGTHS[length]
    seed, g = get_seed(seed); t0 = time.time()
    black, white = Image.new("L", (w, h), 0), Image.new("L", (w, h), 255)
    video = [fit(img, h, w)] + [Image.new("RGB", (w, h), GRAY)] * (n - 1)
    mask = [black] + [white] * (n - 1)
    if last_img is not None:
        video[-1] = fit(last_img, h, w); mask[-1] = black
    use("vace"); st, cf = qset(quality)
    frames = vpipe(video=video, mask=mask, prompt_embeds=encode(full), negative_prompt_embeds=encode(NEG),
                   height=h, width=w, num_frames=n, num_inference_steps=st,
                   guidance_scale=cf, generator=g).frames[0]
    path = new_path("i2v", seed); export_to_video(frames, path, fps=16)
    return done(path, t0, seed, note)

# ---------- 3) Extend video ----------
def gen_extend(src, prompt, style, length, quality, seed, progress=gr.Progress(track_tqdm=True)):
    if not src:
        raise gr.Error("Video chuniye (ya 'Last video lo' dabaiye)")
    full, note = prep_prompt(prompt, style)
    old = [f.convert("RGB") for f in load_video(src)]
    ow, oh = old[0].size
    h, w = RATIOS[auto_ratio(old[0])]
    old = [fit(f, h, w) for f in old] if (ow, oh) != (w, h) else old
    n = LENGTHS[length]; seed, g = get_seed(seed); t0 = time.time()
    black, white = Image.new("L", (w, h), 0), Image.new("L", (w, h), 255)
    video = old[-KEEP:] + [Image.new("RGB", (w, h), GRAY)] * (n - KEEP)
    mask = [black] * KEEP + [white] * (n - KEEP)
    use("vace"); st, cf = qset(quality)
    new = vpipe(video=video, mask=mask, prompt_embeds=encode(full), negative_prompt_embeds=encode(NEG),
                height=h, width=w, num_frames=n, num_inference_steps=st,
                guidance_scale=cf, generator=g, output_type="pil").frames[0]
    path = new_path("ext", seed)
    export_to_video(old + list(new[KEEP:]), path, fps=16)
    return done(path, t0, seed, f"{note}\n\nKul lambai: {(len(old)+len(new)-KEEP)/16:.1f} sec")


# ---------- 4) Lamba video (10 sec - 1 min, ek click) ----------
LONG = {"10 sec": 10, "15 sec": 15, "30 sec": 30, "60 sec (bahut slow)": 60}
def gen_long(img, prompts, style, ratio, total, quality, seed, progress=gr.Progress(track_tqdm=True)):
    lines = [l.strip() for l in (prompts or "").splitlines() if l.strip()]
    if not lines:
        raise gr.Error("Kam se kam ek line prompt likhiye")
    eng = [to_english(l)[0] + STYLES[style] for l in lines]
    seed, g = get_seed(seed); t0 = time.time()
    h, w = RATIOS[auto_ratio(img)] if img is not None else RATIOS[ratio]
    (steps, cf), n, need = qset(quality), 81, LONG[total] * 16
    black, white = Image.new("L", (w, h), 0), Image.new("L", (w, h), 255)
    neg = encode(NEG); path = new_path("long", seed); frames = []; k = 0
    while len(frames) < need:
        emb = encode(eng[min(k, len(eng) - 1)])
        if k == 0 and img is None:
            use("t2v")
            new = pipe(prompt_embeds=emb, negative_prompt_embeds=neg, height=h, width=w, num_frames=n,
                       num_inference_steps=steps, guidance_scale=cf, generator=g, output_type="pil").frames[0]
            frames = list(new)
        else:
            keep = 1 if k == 0 else KEEP
            start = [fit(img, h, w)] if k == 0 else frames[-KEEP:]
            video = start + [Image.new("RGB", (w, h), GRAY)] * (n - keep)
            mask = [black] * keep + [white] * (n - keep)
            use("vace")
            new = vpipe(video=video, mask=mask, prompt_embeds=emb, negative_prompt_embeds=neg,
                        height=h, width=w, num_frames=n, num_inference_steps=steps,
                        guidance_scale=cf, generator=g, output_type="pil").frames[0]
            frames = list(new) if k == 0 else frames + list(new[KEEP:])
        k += 1
        export_to_video(frames[:need], path, fps=16)  # har hisse ke baad save
        gc.collect(); torch.cuda.empty_cache()
        yield path, f"Hissa {k} ready | {len(frames[:need])/16:.1f}/{LONG[total]} sec | {int(time.time()-t0)//60} min", history()
    yield path, f"Pura ban gaya! {LONG[total]} sec | {int(time.time()-t0)//60} min | seed {seed}", history()

def last_video():
    h = history()
    return h[0] if h else None

CSS = """
.gradio-container {max-width: 560px !important; margin: auto;}
#title {text-align:center;}
.gen {background: linear-gradient(90deg,#7b2ff7,#f107a3) !important; color:white !important; font-size:18px;}
"""

def common(default_len="2 sec (fast)"):
    style = gr.Radio(list(STYLES), value="Cinematic", label="Style")
    length = gr.Radio(list(LENGTHS), value=default_len, label="Length")
    quality = gr.Radio(list(QUALITY), value=DEFAULT_Q, label="Quality")
    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
    return style, length, quality, seed

with gr.Blocks(title="AI Video Studio") as demo:
    gr.Markdown("## 🎬 AI Video Studio\nHindi ya English, dono mein prompt likh sakte hain.", elem_id="title")
    with gr.Tab("Text→Video"):
        p1 = gr.Textbox(label="Prompt", lines=3, placeholder="सूर्यास्त में पहाड़ों पर दौड़ती लाल कार")
        s1, l1, q1, sd1 = common()
        r1 = gr.Radio(list(RATIOS), value="9:16", label="Size")
        b1 = gr.Button("✨ Generate", elem_classes="gen")
        o1 = gr.Video(label="Result", autoplay=True); i1 = gr.Markdown()
    with gr.Tab("Photo→Video"):
        img = gr.Image(type="pil", label="Photo (pehla frame)")
        last = gr.Image(type="pil", label="Aakhri photo (optional)")
        p2 = gr.Textbox(label="Kya hona chahiye?", lines=2, placeholder="तांत्रिक धीरे से आँखें खोलता है, पीछे बिजली चमकती है")
        s2, l2, q2, sd2 = common("3 sec")
        b2 = gr.Button("✨ Photo ko chalao", elem_classes="gen")
        o2 = gr.Video(label="Result", autoplay=True); i2 = gr.Markdown()
    with gr.Tab("Extend"):
        src = gr.Video(label="Video jise aage badhana hai")
        lastbtn = gr.Button("Last video lo")
        p3 = gr.Textbox(label="Aage kya hoga?", lines=2)
        s3, l3, q3, sd3 = common("3 sec")
        b3 = gr.Button("➕ Video lamba karo", elem_classes="gen")
        o3 = gr.Video(label="Result", autoplay=True); i3 = gr.Markdown()
    with gr.Tab("Lamba Video"):
        gr.Markdown("Har line = agla ~5 sec ka scene. Ek line likhoge to wahi poore video mein chalegi.")
        img4 = gr.Image(type="pil", label="Shuruaati photo (optional)")
        p4 = gr.Textbox(label="Scenes (ek line = ek scene)", lines=5,
                        placeholder="तांत्रिक श्मशान में ध्यान में बैठा है\nवह आँखें खोलता है, बिजली चमकती है\nवह खड़ा होकर त्रिशूल उठाता है")
        t4 = gr.Radio(list(LONG), value="10 sec", label="Kul lambai")
        r4 = gr.Radio(list(RATIOS), value="16:9", label="Size (photo ho to photo jaisa)")
        s4 = gr.Radio(list(STYLES), value="Mythic Dark", label="Style")
        q4 = gr.Radio(list(QUALITY), value=DEFAULT_Q, label="Quality")
        sd4 = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
        b4 = gr.Button("🎞️ Lamba video banao", elem_classes="gen")
        o4 = gr.Video(label="Result", autoplay=True); i4 = gr.Markdown()
    with gr.Tab("My Videos"):
        refresh = gr.Button("Refresh")
        gallery = gr.Files(label="Saare videos (tap karke download)", value=history())

    b1.click(gen_text, [p1, s1, r1, l1, q1, sd1], [o1, i1, gallery])
    b2.click(gen_image, [img, last, p2, s2, l2, q2, sd2], [o2, i2, gallery])
    b3.click(gen_extend, [src, p3, s3, l3, q3, sd3], [o3, i3, gallery])
    b4.click(gen_long, [img4, p4, s4, r4, t4, q4, sd4], [o4, i4, gallery])
    lastbtn.click(last_video, None, src)
    refresh.click(history, None, gallery)

THEME = gr.themes.Soft(primary_hue="purple")
try:
    demo.queue(max_size=10).launch(share=True, debug=False, theme=THEME, css=CSS)
except TypeError:
    demo.queue(max_size=10).launch(share=True, debug=False)
# 'Running on public URL: https://xxxx.gradio.live' -> wo link phone me kholiye
