# 设置CUDA设备
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定使用GPU 1

# 导入必要的库
import gc 
import re
import time
from dataclasses import dataclass  # 用于创建数据类
from glob import iglob
import argparse  # 命令行参数解析
import torch
from einops import rearrange  # 张量操作库
from fire import Fire  # 命令行工具
from PIL import ExifTags, Image  # 图像处理
import numpy as np
from torchvision import transforms  # 图像变换
from torch.utils.data import DataLoader  # 数据加载
from tqdm import tqdm  # 进度条

# 导入flux模型相关模块
from flux.sampling import denoise_midpoint, denoise_fireflow, denoise_rf_solver, denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5, save_velocity_distribution)
from transformers import pipeline  # HuggingFace模型管道
from datasets import get_dataloader  # 数据集加载
from utils.utils import *
from utils.metrics import *  # 评估指标

NSFW_THRESHOLD = 0.85  # NSFW内容检测阈值

# 定义采样选项的数据类
@dataclass
class SamplingOptions:
    source_prompt: str  # 源图像描述
    target_prompt: str  # 目标编辑描述
    width: int  # 图像宽度
    height: int  # 图像高度
    num_steps: int  # 采样步数
    guidance: float  # 引导尺度
    seed: int | None  # 随机种子

# 自定义图像转张量的转换类（不进行归一化）
class ToTensorWithoutScaling(object):
    """将PIL图像转换为张量，保持0-255范围"""
    def __call__(self, pic):
        if pic is None:
            raise ValueError("输入图像为空")
        if isinstance(pic, Image.Image):
            # 确保图像已加载
            pic.load()
            # 转换为字节张量
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # 调整形状为HWC
            img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
            # 转为CHW格式
            img = img.permute((2, 0, 1)).contiguous()
            return img.to(dtype=torch.uint8)
        else:
            raise TypeError(f"输入类型错误: {type(pic)}")

# 张量转PIL图像函数
def tensor_to_pil_uint8(tensor: torch.Tensor) -> Image.Image:
    """将张量转换回PIL图像"""
    tensor = tensor.detach().cpu()  # 移到CPU并断开梯度
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)  # CHW → HWC
    elif tensor.dim() == 4:
        tensor = tensor.squeeze(0).permute(1, 2, 0)  # 去除批次维度
    return Image.fromarray(tensor.numpy())  # 转为PIL图像

# 图像编码函数（转为潜在空间表示）
@torch.inference_mode()  # 禁用梯度计算
def encode(init_image, torch_device, ae):
    """将图像编码到潜在空间"""
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1  # 归一化到[-1,1]
    init_image = init_image.unsqueeze(0)  # 添加批次维度
    init_image = init_image.to(torch_device)  # 移到指定设备
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)  # 编码并转为bfloat16
    return init_image

# 主函数
@torch.inference_mode()
def main(
    args,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",  # 默认使用CUDA
    loop: bool = False,  # 是否循环交互
    offload: bool = False,  # 是否将模型卸载到CPU
    add_sampling_metadata: bool = True,  # 是否添加元数据
):
    """
    使用flux模型批量处理图像进行编辑和评估
    
    参数:
        args: 命令行参数
        device: PyTorch设备(cpu/cuda)
        loop: 是否启动交互式会话
        offload: 是否将模型卸载到CPU以节省内存
        add_sampling_metadata: 是否将提示添加到图像Exif元数据
    """
    torch.set_grad_enabled(False)  # 禁用梯度计算
    
    # 初始化参数
    name = args.name  # 模型名称
    guidance = args.guidance  # 引导尺度
    output_dir = args.output_dir  # 输出目录
    num_steps = args.num_steps  # 步数
    offload = args.offload  # 是否卸载模型
    prefix = args.output_prefix  # 输出前缀
    inject = args.inject  # 特征共享步数
    start_layer_index = args.start_layer_index  # 起始层索引
    end_layer_index = args.end_layer_index  # 结束层索引
    seed = args.seed if args.seed > 0 else None  # 随机种子
    
    # 初始化NSFW分类器
    nsfw_classifier = pipeline("image-classification", 
                             model="Falconsai/nsfw_image_detection", 
                             device=device)

    # 检查模型名称是否有效
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"未知模型名称: {name}, 可选: {available}")

    torch_device = torch.device(device)
    # 设置默认步数
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # 初始化所有组件
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)  # 文本编码器
    clip = load_clip(torch_device)  # CLIP模型
    model = load_flow_model(name, device="cpu" if offload else torch_device)  # 流模型
    ae = load_ae(name, device="cpu" if offload else torch_device)  # 自编码器

    # 如果需要卸载模型
    if offload:
        model.cpu()
        torch.cuda.empty_cache()  # 清空CUDA缓存
        ae.encoder.to(torch_device)  # 只保留编码器在GPU上

    # 准备数据转换
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),  # 调整大小
        ToTensorWithoutScaling()  # 转为张量(不归一化)
    ])
    
    # 获取数据加载器
    dataset = get_dataloader(
        args.eval_dataset,  # 评估数据集路径
        transform  # 应用的变换
    )
    if dataset is None:
        raise ValueError(f"Failed to load dataset from {args.dataset_path}. Please check the path and try again.")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,  # 批大小
        shuffle=False,  # 不随机打乱
        num_workers=args.num_workers  # 工作线程数
    )

    # 初始化评估指标
    metrics = {
        'clip_score': 0.0,  # CLIP相似度(图像-文本)
        'clip_score_i': 0.0,  # CLIP相似度(图像-图像)
        'mse': 0.0,  # 均方误差
        'psnr': 0.0,  # 峰值信噪比
        'lpips': 0.0,  # 感知相似度
        'ssim': 0.0,  # 结构相似性
        'dino': 0.0,  # DINO特征相似度
        'count': 0  # 样本计数
    }
    

    # 处理循环(带进度条)
    progress = tqdm(dataloader, desc="Processing")
    for batch_idx, (images, source_prompts, target_prompts) in enumerate(progress):
        # 初始化批次指标
        batch_metrics = {
            'clip_score': 0.0,
            'clip_score_i': 0.0,
            'mse': 0.0,
            'psnr': 0.0,
            'lpips': 0.0,
            'ssim': 0.0,
            'dino': 0.0,
            'count': 0
        }
        
        # 处理批次中的每个样本
        for idx in range(len(images)):
            #try:
                # 1. 准备输入图像 -------------------------------
                # 将张量转为PIL图像
                print(images.shape)
                print(images[idx].shape)
                init_image_pil = tensor_to_pil_uint8(images[idx])
                # 转为numpy数组
                init_image_array = np.array(init_image_pil)
                
                # 调整尺寸为16的倍数(模型要求)
                shape = init_image_array.shape
                new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
                new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
                init_image = init_image_array[:new_h, :new_w, :]
                width, height = init_image.shape[0], init_image.shape[1]
                
                # 编码图像到潜在空间
                if offload:
                    ae.encoder.to(torch_device)
                    torch.cuda.empty_cache()
                init_image = encode(init_image, torch_device, ae)
                if offload:
                    ae.encoder.cpu()
                    torch.cuda.empty_cache()

                # 2. 设置采样选项 -------------------------------
                rng = torch.Generator(device="cpu")
                opts = SamplingOptions(
                    source_prompt=source_prompts[idx],  # 源提示
                    target_prompt=target_prompts[idx],  # 目标提示
                    width=width,  # 图像宽度
                    height=height,  # 图像高度
                    num_steps=num_steps,  # 采样步数
                    guidance=guidance,  # 引导尺度
                    seed=seed,  # 随机种子
                )

                # 3. 模型卸载管理 -------------------------------
                if offload:
                    ae = ae.cpu()
                    torch.cuda.empty_cache()
                    t5, clip = t5.to(torch_device), clip.to(torch_device)

                # 4. 准备特征控制参数 ---------------------------
                info = {
                    'feature_path': args.feature_path,  # 特征保存路径
                    'feature': {},  # 特征字典
                    'inject_step': inject,  # 特征注入步数
                    'start_layer_index': start_layer_index,  # 起始层
                    'end_layer_index': end_layer_index,  # 结束层
                    'reuse_v': args.reuse_v,  # 是否重用V
                    'editing_strategy': args.editing_strategy,  # 编辑策略
                    'qkv_ratio': list(map(float, args.qkv_ratio.split(',')))  # QKV比例
                }

                # 5. 准备模型输入 -------------------------------
                # 源提示编码
                inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
                # 目标提示编码
                inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
                # 获取时间步调度
                timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

                # 6. 模型加载管理 -------------------------------
                if offload:
                    t5, clip = t5.cpu(), clip.cpu()
                    torch.cuda.empty_cache()
                    model = model.to(torch_device)
                
                # 7. 选择去噪策略 -------------------------------
                denoise_strategies = {
                    'reflow': denoise,  # 常规流
                    'rf_solver': denoise_rf_solver,  # RF求解器
                    'fireflow': denoise_fireflow,  # 快速流
                    'rf_midpoint': denoise_midpoint,  # 中点法
                }
                if args.sampling_strategy not in denoise_strategies:
                    raise Exception("未知去噪策略")
                denoise_strategy = denoise_strategies[args.sampling_strategy]

                # 8. 反转过程(编码) -----------------------------
                z, info = denoise_strategy(model, **inp, timesteps=timesteps, 
                                          guidance=1, inverse=True, info=info)
                inp_target["img"] = z  # 使用反转后的潜在表示

                # 9. 去噪过程(解码) -----------------------------
                timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], 
                                       shift=(name != "flux-schnell"))
                x, _ = denoise_strategy(model, **inp_target, timesteps=timesteps, 
                                      guidance=guidance, inverse=False, info=info)
                
                # 10. 模型卸载管理 ------------------------------
                if offload:
                    model.cpu()
                    torch.cuda.empty_cache()
                    ae.decoder.to(x.device)

                # 11. 解码潜在表示到像素空间 ---------------------
                batch_x = unpack(x.float(), opts.width, opts.height)
                x = batch_x[0].unsqueeze(0)  # 取第一个样本并添加批次维度
                
                # 使用自动混合精度解码
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                # 同步CUDA操作
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # 12. 后处理输出图像 ---------------------------
                # 裁剪到[-1,1]范围
                x = x.clamp(-1, 1)
                # 嵌入水印
                x = embed_watermark(x.float())
                # 重排维度 HWC
                x = rearrange(x[0], "c h w -> h w c")
                # 转为PIL图像(0-255)
                edited_img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

                # 13. 准备指标计算 ----------------------------
                # 编辑后图像转为张量
                # 首先应用变换将PIL图像转为tensor
                transformed = transforms.Compose([
                    transforms.Resize((args.height, args.width)),  # 调整大小
                    transforms.ToTensor(),  # 转为张量
                    transforms.Normalize([0.5], [0.5])  # 归一化
                ])(edited_img)
                
                # 然后对tensor应用unsqueeze操作
                edited_tensor = transformed.unsqueeze(0).to(device)

                # 原始图像转为张量
                orig_tensor = transforms.Compose([
                    transforms.Resize((args.height, args.width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])(init_image_pil).unsqueeze(0).to(device)

                # 14. 计算评估指标 ----------------------------
                # CLIP分数(文本-图像对齐度)
                if offload:
                    model.cpu()
                    torch.cuda.empty_cache()               
                metric_calculator = metircs()  # 指标计算器
                # CLIP分数(图像-文本对齐度)
                clip_score = metric_calculator.clip_scores(edited_tensor, target_prompts[idx])
                # CLIP分数(图像-图像对齐度)
                clip_score_i = metric_calculator.clip_scores(edited_tensor, orig_tensor)
                # 均方误差
                mse = metric_calculator.mse_scores(edited_tensor, orig_tensor)
                # 峰值信噪比
                psnr_val = metric_calculator.psnr_scores(edited_tensor, orig_tensor)
                # 感知相似度
                lpips_val = metric_calculator.lpips_scores(edited_tensor, orig_tensor)
                # 结构相似性
                ssim_val = metric_calculator.ssim_scores(edited_tensor, orig_tensor)
                # DINO特征相似度
                dino_val = metric_calculator.dino_scores(edited_tensor, orig_tensor)


                # 显式删除中间张量
                del z, x, batch_x, transformed, init_image, metric_calculator
                if offload:
                    model.cpu()
                    t5.cpu()
                    clip.cpu()
                    ae.cpu()
                    torch.cuda.empty_cache()

                # 15. 打印样本指标 ---------------------------
                print(f"\n样本 {batch_idx}-{idx} 指标:")
                print(f"源提示: {source_prompts[idx]}")
                print(f"目标提示: {target_prompts[idx]}")
                print(f"CLIP-T分数: {clip_score:.4f}")
                print(f"CLIP-I分数: {clip_score_i:.4f}")
                print(f"均方误差: {mse:.4f}")
                print(f"峰值信噪比: {psnr_val:.4f} dB")
                print(f"感知相似度: {lpips_val:.4f}")
                print(f"结构相似性: {ssim_val:.4f}")
                print(f"DINO相似度: {dino_val:.4f}")

                # 16. 更新批次指标 ---------------------------
                batch_metrics['clip_score'] += clip_score
                batch_metrics['clip_score_i'] += clip_score_i
                batch_metrics['mse'] += mse
                batch_metrics['psnr'] += psnr_val
                batch_metrics['lpips'] += lpips_val
                batch_metrics['ssim'] += ssim_val
                batch_metrics['dino'] += dino_val
                batch_metrics['count'] += 1

                # 17. 保存结果(如果需要) ----------------------
                if args.save_samples:
                    # 检查NSFW内容
                    nsfw_score = [x["score"] for x in nsfw_classifier(edited_img) 
                                if x["label"] == "nsfw"][0]
                    if nsfw_score < NSFW_THRESHOLD:
                        # 创建输出路径
                        output_name = os.path.join(output_dir, f"{prefix}_img_{batch_idx}_{idx}.jpg")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        
                        # 添加EXIF元数据
                        exif_data = Image.Exif()
                        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                        exif_data[ExifTags.Base.Model] = name
                        if add_sampling_metadata:
                            exif_data[ExifTags.Base.ImageDescription] = source_prompts[idx]
                        # 保存图像
                        edited_img.save(output_name, exif=exif_data, quality=95, subsampling=0)
                    else:
                        print("生成图像可能包含NSFW内容")

            # except Exception as e:
            #     print(f"处理样本 {batch_idx}-{idx} 时出错: {str(e)}")
            #     continue

        # 18. 打印批次摘要 --------------------------------
        print(f"\n{'='*40}")
        print(f"批次 {batch_idx} 指标摘要:")
        
        # 计算批次平均值并更新总体指标
        for key in ['clip_score','clip_score_i', 'mse', 'psnr', 'lpips', 'ssim', 'dino']:
            if batch_metrics['count'] > 0:
                batch_avg = batch_metrics[key] / batch_metrics['count']  # 批次平均值
                metrics[key] += batch_avg  # 累加到总指标
                
                print(f"[{key.upper()}]:")
                print(f"  批次平均: {batch_avg:.4f}")
                print(f"  累计总值: {metrics[key]:.4f}")
                print("-"*30)
        
        # 更新总样本数
        metrics['count'] += batch_metrics['count']
        
        # 清理显存
        del images, source_prompts, target_prompts  # 删除输入数据引用
        if offload:
            # 确保所有模型组件在CPU
            model = model.cpu()
            t5 = t5.cpu()
            clip = clip.cpu()
            ae = ae.cpu()
        # 强制释放所有未使用的缓存
        torch.cuda.empty_cache()
        # 显式调用垃圾回收
        gc.collect()
        print(f"已清理显存，当前使用量: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # 19. 计算最终指标 --------------------------------
    if metrics['count'] > 0:
        print("\n最终评估指标:")
        for key in ['clip_score','clip_score_i', 'mse', 'psnr', 'lpips', 'ssim', 'dino']:
            final_metric = metrics[key] / metrics['count']  # 计算平均值
            print(f"{key.upper():<10} | {final_metric:.4f}")  # 格式化输出

    # 20. 清理资源 ---------------------------------
    del t5, clip, model, ae  # 删除模型
    torch.cuda.empty_cache()  # 清空CUDA缓存

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='RF-Edit批量处理')
    
    # 模型参数
    parser.add_argument('--name', default='flux-dev', type=str,
                      help='flux模型名称')
    parser.add_argument('--eval_dataset', type=str,default='', help='选择要编辑的数据集: EditEval_v1, PIE-Bench')
    
    # 特征控制参数
    parser.add_argument('--feature_path', type=str, default='feature',
                      help='特征保存路径')
    parser.add_argument('--guidance', type=float, default=5,
                      help='引导尺度')
    parser.add_argument('--num_steps', type=int, default=25,
                      help='反转和去噪的步数')
    parser.add_argument('--inject', type=int, default=20,
                      help='应用特征共享的步数')
    parser.add_argument('--start_layer_index', type=int, default=20,
                      help='开始特征共享的层索引')
    parser.add_argument('--end_layer_index', type=int, default=37,
                      help='结束特征共享的层索引')
    
    # 输出参数
    parser.add_argument('--output_dir', default='output', type=str,
                      help='编辑图像输出路径')
    parser.add_argument('--output_prefix', default='editing', type=str,
                      help='编辑图像前缀名')
    parser.add_argument('--save_samples', action='store_true',
                      help='是否保存单个样本图像')
    
    # 处理参数
    parser.add_argument('--sampling_strategy', default='rf_solver', type=str,
                      help='推理时使用的采样方法')
    parser.add_argument('--offload', action='store_true', 
                      help='设为True可节省GPU内存')
    parser.add_argument('--reuse_v', type=int, default=1,
                      help='在反转和重建/编辑期间是否重用V')
    parser.add_argument('--editing_strategy', default='replace_v', type=str,
                      help='编辑策略')
    parser.add_argument('--qkv_ratio', type=str, default='1.0,1.0,1.0', 
                      help='QKV比例，逗号分隔的浮点数')
    parser.add_argument('--seed', type=int, default=0,
                      help='随机种子')
    
    # 数据参数
    parser.add_argument('--height', type=int, default=512,
                      help='输出图像高度')
    parser.add_argument('--width', type=int, default=512,
                      help='输出图像宽度')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='处理批大小')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载工作线程数')

    # 解析参数并运行主函数
    args = parser.parse_args()
    main(args)