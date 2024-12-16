# 使用三阶 Runge-Kutta 方法进行图像反演
with pipeline.progress_bar(total=len(timesteps) - 1) as progress_bar:
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        h = t_prev - t_curr  # 时间步长
        t_vec = torch.full((packed_latents.shape[0],), t_curr, dtype=packed_latents.dtype, device=packed_latents.device)

        joint_attention_kwargs['t'] = t_prev
        joint_attention_kwargs['inverse'] = True
        joint_attention_kwargs['second_order'] = False
        joint_attention_kwargs['inject'] = inject_list[i]
        # print("joint_attention_kwargs id:",id(joint_attention_kwargs))
        # 第一步：计算 k1
        k1, joint_attention_kwargs = pipeline.transformer(
            hidden_states=packed_latents,
            timestep=t_vec,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            # TODO:此处可以传递inject的相关参数，详细仍需再研究，考虑形如 joint_attention_kwargs['inject'] = inject_list[i]  24/11/19 修改到此
            return_dict=pipeline,
        )
        k1 = k1[0]
        # print("joint_attention_kwargs id:",id(joint_attention_kwargs))

        # 第二步：计算 k2
        packed_latents_k2 = packed_latents + 0.5 * h * k1
        t_k2 = t_curr + 0.5 * h
        t_vec_k2 = torch.full((packed_latents.shape[0],), t_k2, dtype=packed_latents.dtype,
                              device=packed_latents.device)
        k2, joint_attention_kwargs = pipeline.transformer(
            hidden_states=packed_latents_k2,
            timestep=t_vec_k2,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=pipeline,
        )
        k2 = k2[0]

        # 第三步：计算 k3
        packed_latents_k3 = packed_latents - h * k1 + 2 * h * k2
        t_k3 = t_curr + h
        t_vec_k3 = torch.full((packed_latents.shape[0],), t_k3, dtype=packed_latents.dtype,
                              device=packed_latents.device)
        joint_attention_kwargs['second_order'] = False
        # joint_attention_kwargs['third_order'] = True  # 标记为三阶计算
        k3, joint_attention_kwargs = pipeline.transformer(
            hidden_states=packed_latents_k3,
            timestep=t_vec_k3,
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=pipeline,
        )
        k3 = k3[0]

        # 防止精度问题
        packed_latents = packed_latents.to(torch.float32)
        k1 = k1.to(torch.float32)
        k2 = k2.to(torch.float32)
        k3 = k3.to(torch.float32)
        # 更新潜变量
        packed_latents = packed_latents + (h / 6) * (k1 + 4 * k2 + k3)