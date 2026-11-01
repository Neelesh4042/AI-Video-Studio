# AI Video Studio (personal use)

Wan2.1 1.3B (Text→Video) + Wan2.1-VACE 1.3B (Photo→Video, Extend, Lamba Video) on Kaggle T4 x2, Gradio UI for phone. Hindi prompts supported.

## Kaggle cell (GPU T4 x2 + Internet on)

```python
!pip install -q -U diffusers transformers accelerate peft gradio ftfy imageio imageio-ffmpeg sentencepiece deep-translator
!rm -rf /tmp/AVS && git clone -q https://github.com/Neelesh4042/AI-Video-Studio /tmp/AVS
exec(open("/tmp/AVS/studio.py").read())
```

Har naye version ke liye: Run → Restart & clear outputs, phir yahi cell chalao. Latest code apne aap aa jayega.

Videos: `/kaggle/working/videos`
