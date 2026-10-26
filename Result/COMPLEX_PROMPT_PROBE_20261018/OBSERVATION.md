# Complex prompt observation pair

T50: original Wan2.1-T2V-14B, 50-step UniPC, CFG=5, shift=5.

S4: LightX2V bidirectional four-step distilled model, CFG disabled, shift=5.

Both: 832x480, 81 frames, 16 FPS, identical source noise and positive text condition.

Seed: 2026101801. First fixed pair; no seed search or selection.

This pair is for user observation only. No FID/FVD, human rating, qualification claim or S3 correction.

English prompt:

A single continuous five-second, eye-level wide shot inside a sunlit pottery workshop. Exactly two adults stand behind an oak table: a man wearing a white linen shirt on the left and a woman wearing a dark blue apron on the right. On the table are a red ceramic teapot at the center, a blue cup to its left, a yellow cup to its right, and an empty rectangular wooden tray near the right edge. During the first three seconds, the man lifts the red teapot with his right hand and tilts it to pour a thin, continuous stream of tea into the blue cup while the woman steadies that cup with her left hand. He then stops pouring and returns the teapot upright to its original position. Only after the pouring has stopped, the woman slides the blue cup onto the wooden tray; the yellow cup stays in place throughout. At the same time, a small gray cat walks from left to right on the floor beneath the table, briefly disappears behind one table leg, and reappears on the other side. The camera makes a slow, smooth sideways move to the right without cuts or zooms. Maintain consistent faces, clothing, object colors and object counts. Show natural hand grips, continuous object motion, realistic contact with the table, and coherent perspective and shadows.

中文说明：

五秒单镜头、平视广角，阳光照入陶艺工作室。左侧白色亚麻衬衫男子与右侧深蓝围裙女子站在橡木桌后。红色茶壶居中，蓝杯在其左、黄杯在其右，空长方木托盘在桌面右边。前三秒男子用右手提壶向蓝杯倒茶，女子左手扶杯；男子停止倒茶、将壶放回原位后，女子再把蓝杯滑到托盘上，黄杯始终不动。同时灰猫从桌下左向右走，短暂被一条桌腿遮挡再出现。镜头缓慢向右平移，不切镜、不缩放；人物、服饰、物体颜色与数量保持一致，动作、接触、透视和阴影连贯。

- T50: `/data/chenjiayu/changzihe/DistillPP_Validation/Evaluation_Videos/complex_prompt_probe_20261018/T50/CP01_seed2026101801.mp4`
- S4: `/data/chenjiayu/changzihe/DistillPP_Validation/Evaluation_Videos/complex_prompt_probe_20261018/S4/CP01_seed2026101801.mp4`
