"""
原始版本
"""
import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0
):
    #mine三阶
    # # this is ignored for schnell
    # inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    #
    # if inverse:
    #     timesteps = timesteps[::-1]
    #     inject_list = inject_list[::-1]
    # guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    #
    # step_list = []
    # #lyw
    # # for i, (t_curr, t_prev, t_next) in enumerate(zip(timesteps[:-2], timesteps[1:-1], timesteps[2:])):
    # #lyw
    # for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
    #     t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
    #     info['t'] = t_prev if inverse else t_curr
    #     info['inverse'] = inverse
    #     info['second_order'] = False
    #     info['third_order'] = False
    #     info['inject'] = inject_list[i]
    #
    #     pred, info = model(
    #         img=img,
    #         img_ids=img_ids,
    #         txt=txt,
    #         txt_ids=txt_ids,
    #         y=vec,
    #         timesteps=t_vec,
    #         guidance=guidance_vec,
    #         info=info
    #     )
    #     #原来
    #     img_mid = img + (t_prev - t_curr) / 2 * pred
    #     img_14 = img + (t_prev - t_curr) / 4 *  pred
    #     # 这个img是rf-1-solver的结果
    #     # img = img + (t_prev - t_curr) / 2 * pred
    #     # 这个img是rf-1-solver的结果
    #     t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
    #     info['second_order'] = True
    #     pred_mid, info = model(
    #         img=img_mid,
    #         img_ids=img_ids,
    #         txt=txt,
    #         txt_ids=txt_ids,
    #         y=vec,
    #         timesteps=t_vec_mid,
    #         guidance=guidance_vec,
    #         info=info
    #     )
    #     #first_order是这个点的一阶导
    #     first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
    #     #这个img是rf-2-solver的结果
    #     # img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
    #     #这个img是rf-2-solver的结果
    #     #
    #     t_vec_14 = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 4), dtype=img.dtype, device=img.device)
    #     #lyw
    #     # 新增部分，计算三阶导数近似值
    #
    #     pred_14, info = model(
    #         img=img_14,  # 这里输入图像可以根据实际考虑是否用更新后的img等情况，暂按原始img示意
    #         img_ids=img_ids,
    #         txt=txt,
    #         txt_ids=txt_ids,
    #         y=vec,
    #         timesteps=t_vec_14,
    #         guidance=guidance_vec,
    #         info=info
    #     )
    #     first_order18 = (pred_mid - pred_14) / ((t_prev - t_curr) / 4)
    #     first_order38 = (pred_14 - pred) / ((t_prev - t_curr) / 4)
    #
    #     second_order = (first_order38 - first_order18) / ((t_prev - t_curr) / 4)
    #
    #     # 按照三阶泰勒展开更新img
    #     img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order + (1 / 6) * (
    #                 t_prev - t_curr) ** 3 * second_order
    #     #lyw
    #
    #
    # return img, info
    #mine三阶

    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        #原来
        # img_mid = img + (t_prev - t_curr) / 2 * pred
        #原来
        img = img + (t_prev - t_curr) / 2 * pred
        # t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        # info['second_order'] = True
        # pred_mid, info = model(
        #     img=img_mid,
        #     img_ids=img_ids,
        #     txt=txt,
        #     txt_ids=txt_ids,
        #     y=vec,
        #     timesteps=t_vec_mid,
        #     guidance=guidance_vec,
        #     info=info
        # )
        #
        # first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        # img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
"""
原始版本
"""
#
# import math
# from typing import Callable
#
# import torch
# from einops import rearrange, repeat
# from torch import Tensor
#
# from .model import Flux
# from .modules.conditioner import HFEmbedder
#
#
# def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
#     bs, c, h, w = img.shape
#     if bs == 1 and not isinstance(prompt, str):
#         bs = len(prompt)
#
#     img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
#     if img.shape[0] == 1 and bs > 1:
#         img = repeat(img, "1 ... -> bs ...", bs=bs)
#
#     img_ids = torch.zeros(h // 2, w // 2, 3)
#     img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
#     img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
#     img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
#
#     if isinstance(prompt, str):
#         prompt = [prompt]
#     txt = t5(prompt)
#     if txt.shape[0] == 1 and bs > 1:
#         txt = repeat(txt, "1 ... -> bs ...", bs=bs)
#     txt_ids = torch.zeros(bs, txt.shape[1], 3)
#
#     vec = clip(prompt)
#     if vec.shape[0] == 1 and bs > 1:
#         vec = repeat(vec, "1 ... -> bs ...", bs=bs)
#
#     return {
#         "img": img,
#         "img_ids": img_ids.to(img.device),
#         "txt": txt.to(img.device),
#         "txt_ids": txt_ids.to(img.device),
#         "vec": vec.to(img.device),
#     }
#
#
# def time_shift(mu: float, sigma: float, t: Tensor):
#     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
#
#
# def get_lin_function(
#     x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
# ) -> Callable[[float], float]:
#     m = (y2 - y1) / (x2 - x1)
#     b = y1 - m * x1
#     return lambda x: m * x + b
#
#
# def get_schedule(
#     num_steps: int,
#     image_seq_len: int,
#     base_shift: float = 0.5,
#     max_shift: float = 1.15,
#     shift: bool = True,
# ) -> list[float]:
#     # extra step for zero
#     timesteps = torch.linspace(1, 0, num_steps + 1)
#
#     # shifting the schedule to favor high timesteps for higher signal images
#     if shift:
#         # estimate mu based on linear estimation between two points
#         mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
#         timesteps = time_shift(mu, 1.0, timesteps)
#
#     return timesteps.tolist()
#
#
# def denoise(
#     model: Flux,
#     # model input
#     img: Tensor,
#     img_ids: Tensor,
#     txt: Tensor,
#     txt_ids: Tensor,
#     vec: Tensor,
#     # sampling parameters
#     timesteps: list[float],
#     inverse,
#     info,
#     guidance: float = 4.0
# ):
#
#     # this is ignored for schnell
#     inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
#
#     if inverse:
#         timesteps = timesteps[::-1]
#         inject_list = inject_list[::-1]
#     guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
#
#     step_list = []
#     for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
#         t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
#         info['t'] = t_prev if inverse else t_curr
#         info['inverse'] = inverse
#         info['second_order'] = False
#         info['third_order'] = False
#         info['inject'] = inject_list[i]
#
#         pred_1, info = model(
#             img=img,
#             img_ids=img_ids,
#             txt=txt,
#             txt_ids=txt_ids,
#             y=vec,
#             timesteps=t_vec,
#             guidance=guidance_vec,
#             info=info
#         )
#         #原来
#         img_mid1 = img + 0.5 * (t_prev - t_curr) * pred_1
#         t_vec_mid1 = torch.full((img.shape[0],), (t_curr + 0.5 * (t_prev - t_curr)), dtype=img.dtype, device=img.device)
#         info['second_order'] = True
#         pred_2, info = model(
#             img=img_mid1,
#             img_ids=img_ids,
#             txt=txt,
#             txt_ids=txt_ids,
#             y=vec,
#             timesteps=t_vec_mid1,
#             guidance=guidance_vec,
#             info=info
#         )
#
#         img_mid2 = img - (t_prev - t_curr) * pred_1 + 2 * (t_prev - t_curr) * pred_2
#
#         #
#         t_vec_mid2 = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr)), dtype=img.dtype, device=img.device)
#         #lyw
#         # 新增部分，计算三阶导数近似值
#
#         pred_3, info = model(
#             img=img_mid2,  # 这里输入图像可以根据实际考虑是否用更新后的img等情况，暂按原始img示意
#             img_ids=img_ids,
#             txt=txt,
#             txt_ids=txt_ids,
#             y=vec,
#             timesteps=t_vec_mid2,
#             guidance=guidance_vec,
#             info=info
#         )
#         img = img + ((t_prev - t_curr) / 6) * (pred_1 + 4 * pred_2 + pred_3)
#         #lyw
#
#
#     return img, info
#
#
#
# def unpack(x: Tensor, height: int, width: int) -> Tensor:
#     return rearrange(
#         x,
#         "b (h w) (c ph pw) -> b c (h ph) (w pw)",
#         h=math.ceil(height / 16),
#         w=math.ceil(width / 16),
#         ph=2,
#         pw=2,
#     )
