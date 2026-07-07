import numpy as np
import torchvision.transforms.functional as TF
import torch

from PIL import Image


def resize_lanczos(img, h, w, method = None):
    img = (img + 1).float().mul_(127.5)
    img = Image.fromarray(np.clip(img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = img.resize((w,h), resample=Image.Resampling.LANCZOS if method is None else method) 
    img = torch.from_numpy(np.array(img).astype(np.float32)).movedim(-1, 0)
    img = img.div(127.5).sub_(1)
    return img

def calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas,  block_size = 16):
    if fit_into_canvas == None or fit_into_canvas == 2:
        # return image_height, image_width
        return canvas_height, canvas_width
    if fit_into_canvas == 1:
        scale1  = min(canvas_height / image_height, canvas_width / image_width)
        scale2  = min(canvas_width / image_height, canvas_height / image_width)
        scale = max(scale1, scale2) 
    else: #0 or #2 (crop)
        scale = (canvas_height * canvas_width / (image_height * image_width))**(1/2)

    new_height = round( image_height * scale / block_size) * block_size
    new_width = round( image_width * scale / block_size) * block_size
    return new_height, new_width

def convert_image_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)

def convert_tensor_to_image(t, frame_no = 0):
    if len(t.shape) == 4:
        t = t[:, frame_no] 
    if t.shape[0]== 1:
        t = t.expand(3,-1,-1)
    if t.dtype == torch.uint8:
        return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())
    
    return Image.fromarray(t.clone().add_(1.).mul_(127.5).permute(1,2,0).to(torch.uint8).cpu().numpy())

def to_rgb_tensor(value, device="cpu", dtype=torch.float):
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        if isinstance(value, (list, tuple, np.ndarray)):
            vals = value
        else:
            vals = [value, value, value]
        tensor = torch.tensor(vals, device=device, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    elif tensor.numel() != 3:
        tensor = tensor.flatten()
        if tensor.numel() < 3:
            tensor = tensor.repeat(3)[:3]
        else:
            tensor = tensor[:3]
    return tensor.view(3, 1, 1)

def fit_image_into_canvas(ref_img, image_size, canvas_tf_bg =127.5, device ="cpu", full_frame = False, outpainting_dims = None, return_mask = False, return_image = False):
    inpaint_color = to_rgb_tensor(canvas_tf_bg, device=device, dtype=torch.float) / 127.5 - 1
    inpaint_color = inpaint_color.unsqueeze(1)

    ref_width, ref_height = ref_img.size
    if (ref_height, ref_width) == image_size and outpainting_dims  == None:
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        canvas = torch.zeros_like(ref_img[:1]) if return_mask else None
    else:
        canvas_height, canvas_width = image_size
        
        if full_frame:
            new_height = canvas_height
            new_width = canvas_width
            top = left = 0 
        else:
            # if fill_max  and (canvas_height - new_height) < 16:
            #     new_height = canvas_height
            # if fill_max  and (canvas_width - new_width) < 16:
            #     new_width = canvas_width
            scale = min(canvas_height / ref_height, canvas_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)
            top = (canvas_height - new_height) // 2
            left = (canvas_width - new_width) // 2
        ref_img = ref_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        
        canvas = inpaint_color.expand(3, 1, canvas_height, canvas_width).clone()
        canvas[:, :, top:top + new_height, left:left + new_width] = ref_img 
        
        ref_img = canvas
        canvas = None
        if return_mask:
            
            canvas = torch.ones((1, 1, canvas_height, canvas_width), dtype= torch.float, device=device) # [-1, 1]
            canvas[:, :, top:top + new_height, left:left + new_width] = 0
            
            canvas = canvas.to(device)
    if return_image:
        return convert_tensor_to_image(ref_img), canvas

    return ref_img.to(device), canvas