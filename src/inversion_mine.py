import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np

import os

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.
    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    torch.set_grad_enabled(False)
    name = args.name
    base_path = args.all_img_dir
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload

    mapping = {'A deserted highway near snow mountains under a partly cloudy sky':
                   'A red car on a deserted highway near snow mountains under a partly cloudy sky',
               'A tranquil lake surrounded by dense forests and mountains under a clear blue sky':
                   'A canoe on the tranquil lake surrounded by dense forests and mountains under a clear blue sky',
               'A plant near a lamp on a table with a wooden clock on the wall':
                   'A book on the table near a plant and a lamp with a wooden clock on the wall',
               'A bench under a tree with fallen leaves on the ground and a historic building in the background':
                   'A doll on the bench under a tree with fallen leaves on the ground and a historic building in the background',
               'An empty room with hardwood floors, white walls, and a large window showing trees outside':
                   'A blue sofa in the center of an room with hardwood floors, white walls, and a large window showing trees outside',
               'An orange cat sitting attentively on a red-painted curb against a white background':
                   'A butterfly near an orange cat sitting attentively on a red-painted curb against a white background',
               'A stack of cookies on a wooden board with a gray background':
                   'A glass of milk next to a stack of cookies on a wooden board with a gray background',
               'A camel walking on a desert': 'A flamingo standing on a camel walking on a desert',
               'A llama toy standing next to a potted plant in a cozy room':
                   'A floor lamp standing next to a potted plant in a cozy room',
               'A wolf stands in a forest':
                   'A rock sits in a forest',
               'A cluster of coconuts attached to a palm tree':
                   'A cluster of lanterns on a palm tree',
               'A person holding a handbag':
                   'A person holding a box',
               'A car with surfboards on top':
                   'A car with bushes on top',
               'A road sign stands beside a rural highway':
                   'A mailbox stands beside the rural highway',
               'A white mug filled with yellow dandelions on a table':
                   'A white mug filled with cute puppies on the table',
               'A red cabin on a rocky outcrop by the sea':
                   'A blue dome on a rocky outcrop by the sea',
               'A wooden bench in front of a lush green hill with trees':
                   'A lush green hill with trees',
               'A bicycle with a basket parked against a rustic wall with vines':
                   'A rustic wall with vines',
               'A cup of coffee on an unmade bed':
                   'An unmade bed',
               'A person sitting at the seaside with rocks':
                   'A seaside with rocks',
               'A cup of coffee and a pair of glasses near an open book':
                   'An open book',
               'A person with an umbrella walking across a busy city street':
                   'A busy city street',
               'A wooden house with a dock on a mountain lake':
                   'A mountain lake',
               'Four women in long dresses walk down a hallway':
                   'A hallway',
               'A man stands on mountains':
                   'A man stands at a seaside',
               'A person sits on a desert':
                   'A person sits in a bedroom',
               'A person stands on a snowy mountain range':
                   'A person stands in a tropical rainforest',
               'A woman leans on a wooden post at a field':
                   'A woman leans on a wooden post at a street',
               'A person sits on a paddleboard on a lake':
                   'A person sits on a paddleboard on a desert',
               'A white fox sitting on a snowland':
                   'A white fox sitting on a beach',
               'Several Rhinoceroses on an African plain':
                   'Several Rhinoceroses on Mars',
               'A silver car parked at a city street':
                   'A silver car parked at a dense jungle',
               'A church on top of a mountain surrounded by trees':
                   'A watercolor painting of a church on top of a mountain surrounded by trees',
               'A light house sitting on a cliff next to the ocean':
                   'A Van Gogh style painting of a light house sitting on a cliff next to the ocean',
               'A cityscape with skyscrapers and a park':
                   'A cyberpunk style of a cityscape with skyscrapers and a park',
               'A countryside field with houses and trees':
                   'A Japanese anime style of a countryside field with houses and trees',
               'A winding road in the middle of a large green landscape':
                   'A winding road in the middle of a large green landscape in winter',
               'A grassy field with yellow flowers near mountains':
                   'A grassy field with yellow flowers near mountains in autumn',
               'A woman standing on a beach next to the ocean':
                   'A woman standing on a beach next to the ocean at sunset',
               'An orange car parked in a parking lot in front of tall buildings':
                   'An orange car parked in a parking lot in front of tall buildings at night',
               'A kitten sitting on a sofa': 'A metal kitten sitting on a sofa',
               'A white horse running in a field':
                   'A statue of a horse running in a field',
               'A chocolate cake on a plate with a fork':
                   'A plastic chocolate cake on a plate with a fork',
               'A person carrying a leather handbag with a fluffy keychain':
                   'A person carrying a glass handbag with a fluffy keychain',
               'A rose in bloom': 'A wooden rose in bloom', 'A black church':
                   'An ice church', 'A white horse standing in the grass':
                   'A white horse running in the grass',
               'A polar bear standing beside the sea':
                   'A polar bear raising its hand',
               'A dog standing on the ground':
                   'A dog jumping on the ground',
               'A bird on the tree': 'A bird is flying over the tree'}

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)
    picture_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    for picture_file in picture_files:
        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {name}, chose from {available}")

        torch_device = torch.device(device)
        if num_steps is None:
            num_steps = 4 if name == "flux-schnell" else 25

        # init all components
        t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
        clip = load_clip(torch_device)
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        ae = load_ae(name, device="cpu" if offload else torch_device)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.encoder.to(torch_device)

        picture_path = os.path.join(base_path, picture_file)
        source_prompt = os.path.splitext(picture_file)[0]  # 图片文件名作为source_prompt和target_prompt
        target_prompt = source_prompt
        init_image = None
        init_image = np.array(Image.open(picture_path).convert('RGB'))

        shape = init_image.shape

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        init_image = init_image[:new_h, :new_w, :]

        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, torch_device, ae)

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if loop:
            opts = parse_prompt(opts)

        while opts is not None:
            if opts.seed is None:
                opts.seed = rng.seed()
            print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
            t0 = time.perf_counter()

            opts.seed = None
            if offload:
                ae = ae.cpu()
                torch.cuda.empty_cache()
                t5, clip = t5.to(torch_device), clip.to(torch_device)

            info = {}
            info['feature_path'] = args.feature_path
            info['feature'] = {}
            info['inject_step'] = args.inject
            if not os.path.exists(args.feature_path):
                os.mkdir(args.feature_path)

            inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

            # offload TEs to CPU, load model to gpu
            if offload:
                t5, clip = t5.cpu(), clip.cpu()
                torch.cuda.empty_cache()
                model = model.to(torch_device)

            # inversion initial noise
            z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)

            inp_target["img"] = z

            timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

            # denoise initial noise
            x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            # decode latents to pixel space
            batch_x = unpack(x.float(), opts.width, opts.height)

            for x in batch_x:
                x = x.unsqueeze(0)
                #lyw
                source_img_name = os.path.basename(picture_path )
                # 构造新的文件名，在原文件名基础上添加reconstructed后缀并保持原扩展名
                output_name = os.path.join(output_dir, source_img_name.rsplit('.', 1)[0] + " reconstructed.jpg")
                #lyw
                # output_name = os.path.join(output_dir, "img_{idx}.jpg")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    idx = 0
                else:
                    fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                    if len(fns) > 0:
                        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                    else:
                        idx = 0

                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                fn = output_name.format(idx=idx)
                print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
                # bring into PIL format and save
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = name
                    if add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = source_prompt
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                    idx += 1
                else:
                    print("Your generated image may contain NSFW content.")

                if loop:
                    print("-" * 80)
                    opts = parse_prompt(opts)
                else:
                    opts = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--all_img_dir', default='../all_pictures_resize', type=str,
                        help='The path of the source image dataset')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=3,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=20,  #原25
                        help='the number of timesteps for inversion and den ising')
    parser.add_argument('--inject', type=int, default=3,     #原20
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='ablation/new_output/output_1_solver_inversion_3_20_3_k_injection', type=str,
                        help='the path of the edited image')
    parser.add_argument('--offload', action='store_true', default=True,help='set it to True if the memory of GPU is not enough')

    args = parser.parse_args()

    main(args)
