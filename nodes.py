import torch
import numpy as np
from PIL import Image
from .style_transfer.style_transfer import StyleTransfer, gen_scales
import comfy.utils
import model_management

def tensor2pil(image):
    # ComfyUI images are (batch, height, width, channels)
    return Image.fromarray(np.clip(255. * image.cpu().numpy()[0], 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class StyleTransferNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content": ("IMAGE",),
                "style": ("IMAGE",),
                "content_weight": ("FLOAT", {"default": 0.015, "min": 0.0, "max": 1.0, "step": 0.001}),
                "style_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "tv_weight": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "min_scale": ("INT", {"default": 128, "min": 16, "max": 2048, "step": 16}),
                "end_scale": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 16}),
                "iterations": ("INT", {"default": 500, "min": 1, "max": 10000, "step": 10}),
                "initial_iterations": ("INT", {"default": 1000, "min": 1, "max": 10000, "step": 10}),
                "step_size": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 1.0, "step": 0.001}),
                "avg_decay": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.001}),
                "init": (["content", "gray", "uniform", "style_stats"],),
                "pooling": (["max", "average", "l2"],),
                "optimizer": (["adam", "lbfgs"],),
                "style_scale_fac": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "style_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stylize"
    CATEGORY = "style_transfer"

    def stylize(self, content, style, content_weight, style_weight, tv_weight, 
                min_scale, end_scale, iterations, initial_iterations, 
                step_size, avg_decay, init, pooling, optimizer,
                style_scale_fac, style_size):
        
        device = model_management.get_torch_device()
        
        # Convert tensors to PIL images
        content_pil = tensor2pil(content)
        style_pils = [Image.fromarray(np.clip(255. * img, 0, 255).astype(np.uint8)) for img in style.cpu().numpy()]
        
        st = StyleTransfer(devices=[device], pooling=pooling)
        
        scales = gen_scales(min_scale, end_scale)
        total_steps = initial_iterations + (len(scales) - 1) * iterations
        pbar = comfy.utils.ProgressBar(total_steps)
        
        def callback(iterate):
            pbar.update(1)

        st_style_size = style_size if style_size > 0 else None

        result_pil = st.stylize(
            content_pil,
            style_pils,
            content_weight=content_weight,
            style_weights=[style_weight] * len(style_pils),
            tv_weight=tv_weight,
            min_scale=min_scale,
            end_scale=end_scale,
            iterations=iterations,
            initial_iterations=initial_iterations,
            step_size=step_size,
            avg_decay=avg_decay,
            init=init,
            optimizer=optimizer,
            style_scale_fac=style_scale_fac,
            style_size=st_style_size,
            callback=callback
        )
        
        return (pil2tensor(result_pil),)

NODE_CLASS_MAPPINGS = {
    "StyleTransfer": StyleTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleTransfer": "Style Transfer (Pytorch)"
}
