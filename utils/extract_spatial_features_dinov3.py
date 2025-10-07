#!/usr/bin/env python3
"""
HEST细胞深度特征提取器
使用DINOv3提取每个细胞的深度特征，结合形态特征构建综合特征向量
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import cv2
from sklearn.decomposition import PCA
import pickle
from tqdm import tqdm
import warnings
import shapely.wkb as wkb
import h5py
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import gc
warnings.filterwarnings("ignore")

# 设置HuggingFace镜像（用于中国网络环境）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f"已设置HuggingFace镜像: {os.environ.get('HF_ENDPOINT')}")

# DINOv3模型导入
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    DINOV3_AVAILABLE = True
    print("✓ DINOv3可用")
except ImportError as e:
    DINOV3_AVAILABLE = False
    print(f"错误: 请安装transformers库: pip install transformers")
    print(f"详细错误: {e}")

class HESTCellFeatureExtractor:
    """HEST细胞深度特征提取器（DINOv3版本）"""
    
    def __init__(self, 
                 hest_data_dir,
                 output_dir,
                 dinov3_model_path=None,      # DINOv3本地模型路径
                 bulk_pca_model_path=None,     # 已移除，仅保留独立PCA
                 cell_patch_size=48,           # 细胞patch大小，基于实际细胞大小分析
                 dinov3_feature_dim=1024,     # DINOv3-L特征维度
                 final_dino_dim=128,          # PCA降维后DINO维度（统一为128）
                 device='cuda',
                 dino_batch_size=256,         # 大幅增加DINOv3批处理大小
                 cell_batch_size=50000,       # 大幅增加细胞批处理大小
                 num_workers=8):              # 多进程工作者数量
        
        if not DINOV3_AVAILABLE:
            raise ImportError("DINOv3不可用，请安装transformers")
            
        self.hest_data_dir = hest_data_dir
        self.output_dir = output_dir
        self.dinov3_model_path = dinov3_model_path
        self.bulk_pca_model_path = bulk_pca_model_path
        self.cell_patch_size = cell_patch_size
        self.dinov3_feature_dim = dinov3_feature_dim
        self.final_dino_dim = final_dino_dim
        self.final_feature_dim = final_dino_dim  # 只使用DINOv3特征，128维
        self.device = device
        self.dino_batch_size = dino_batch_size
        self.cell_batch_size = cell_batch_size
        self.num_workers = num_workers
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化DINOv3模型
        self.init_dinov3_model()
        
        
        print(f"迁移学习配置:")
        print(f"  - DINOv3批处理大小: {self.dino_batch_size}")
        print(f"  - 细胞批处理大小: {self.cell_batch_size}")
        print(f"  - 多进程工作者: {self.num_workers}")
        print(f"  - 设备: {self.device}")
        print(f"  - 最终特征维度: {self.final_feature_dim} (仅DINOv3+PCA)")
        print(f"  - 模型一致性: 使用DINOv3-L")
        
    def init_dinov3_model(self):
        """初始化DINOv3模型（使用torch.hub加载本地权重）"""
        print("初始化DINOv3模型...")
        
        if not self.dinov3_model_path or not os.path.exists(self.dinov3_model_path):
            raise RuntimeError(f"DINOv3模型文件不存在: {self.dinov3_model_path}")
        
        print(f"使用本地DINOv3模型: {self.dinov3_model_path}")
        
        try:
            # 设置 DINOv3 仓库路径
            dinov3_repo_dir = "/data/yujk/hovernet2feature/dinov3"
            
            if not os.path.exists(dinov3_repo_dir):
                raise RuntimeError(f"DINOv3仓库不存在: {dinov3_repo_dir}")
            
            print(f"使用DINOv3仓库: {dinov3_repo_dir}")
            
            # 使用 torch.hub.load 加载 DINOv3 ViT-L/16 模型
            print("使用torch.hub加载DINOv3-ViT-L/16模型...")
            
            # 直接使用torch.hub加载
            self.dino_model = torch.hub.load(
                dinov3_repo_dir, 
                'dinov3_vitl16',  # DINOv3 ViT-L/16 模型
                source='local',
                weights=self.dinov3_model_path,  # 使用本地权重
                trust_repo=True
            )
            
            print("✓ 成功使用torch.hub加载DINOv3模型")
            
            # 设置特征维度
            self.dinov3_feature_dim = 1024  # DINOv3-L 的特征维度
            
            # 设置图像处理器（使用标准的ImageNet预处理）
            from torchvision import transforms
            
            self.dino_processor_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            print("✓ 设置图像预处理器")
            
        except Exception as e:
            print(f"torch.hub加载失败: {e}")
            
            # 备选方案：手动实现DINOv3加载
            print("尝试手动加载...")
            try:
                # 直接加载模型文件
                checkpoint = torch.load(self.dinov3_model_path, map_location='cpu')
                
                # 检查是否是完整的模型对象
                if hasattr(checkpoint, 'forward'):
                    # 直接是模型对象
                    self.dino_model = checkpoint
                    print("✓ 直接加载模型对象")
                else:
                    # 是状态字典，需要先建立模型架构
                    print("检测到状态字典，正在建立模型架构...")
                    
                    # 尝试使用timm或手动建立模型架构
                    try:
                        import timm
                        # 使用timm创廽DINOv3类似的模型
                        self.dino_model = timm.create_model(
                            'vit_large_patch16_224',
                            pretrained=False,
                            num_classes=0,  # 只要特征提取
                            global_pool=''
                        )
                        
                        # 尝试加载权重
                        if isinstance(checkpoint, dict):
                            if 'model' in checkpoint:
                                state_dict = checkpoint['model']
                            elif 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                state_dict = checkpoint
                        else:
                            state_dict = checkpoint
                        
                        # 尝试加载状态字典
                        missing_keys, unexpected_keys = self.dino_model.load_state_dict(state_dict, strict=False)
                        print(f"✓ 使用timm加载模型，缺少: {len(missing_keys)}, 意外: {len(unexpected_keys)}")
                        
                    except ImportError:
                        print("timm不可用，请安装: pip install timm")
                        raise RuntimeError("无法加载DINOv3模型")
                
                # 设置图像处理器
                from torchvision import transforms
                self.dino_processor_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                self.dinov3_feature_dim = 1024
                
            except Exception as e2:
                raise RuntimeError(f"所有DINOv3加载方式都失败:\ntorch.hub: {e}\n手动加载: {e2}")
        
        # 设置模型属性
        self.dino_model.to(self.device)
        
        # 不强制设置为FP16，让autocast自动管理精度
        # if torch.cuda.is_available():
        #     self.dino_model = self.dino_model.half()  # 注释掉强制FP16
        
        self.dino_model.eval()
          
        print(f"✓ DINOv3模型加载成功，特征维度: {self.dinov3_feature_dim}")
    
    def monitor_resources(self):
        """监控系统资源使用情况"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            resource_info = f"CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%"
            
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                resource_info += f", GPU: {gpu_percent:.1f}% ({gpu_memory_used:.2f}/{gpu_memory_total:.1f}GB)"
            
            return resource_info
        except ImportError:
            return "资源监控需要安装psutil: pip install psutil"
    
    def load_sample_data(self, sample_id):
        """加载单个样本的数据"""
        print(f"加载样本数据: {sample_id}")
        
        sample_data = {}
        
        # 加载WSI图像
        wsi_path = os.path.join(self.hest_data_dir, "wsis", f"{sample_id}.tif")
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI文件不存在: {wsi_path}")
        
        # 加载细胞分割数据
        cellvit_path = os.path.join(self.hest_data_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
        if not os.path.exists(cellvit_path):
            raise FileNotFoundError(f"细胞分割文件不存在: {cellvit_path}")
        
        cellvit_df = pd.read_parquet(cellvit_path)
        
        sample_data = {
            'wsi_path': wsi_path,
            'cellvit_df': cellvit_df,
            'sample_id': sample_id
        }
        
        print(f"  WSI: {wsi_path}")
        print(f"  细胞数量: {len(cellvit_df)}")
        
        return sample_data
    
    def extract_cell_patch(self, wsi_image, cell_geometry, patch_size=None):
        """从WSI中提取单个细胞的图像patch"""
        if patch_size is None:
            patch_size = self.cell_patch_size
        
        try:
            # 解析细胞几何形状
            if isinstance(cell_geometry, bytes):
                geom = wkb.loads(cell_geometry)
            else:
                geom = cell_geometry
            
            # 获取细胞边界框
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            
            # 计算细胞中心点和patch区域
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            # 定义patch区域
            half_size = patch_size // 2
            x1 = max(0, int(center_x - half_size))
            y1 = max(0, int(center_y - half_size))
            x2 = min(wsi_image.shape[1], int(center_x + half_size))
            y2 = min(wsi_image.shape[0], int(center_y + half_size))
            
            # 提取patch
            cell_patch = wsi_image[y1:y2, x1:x2]
            
            # 如果patch大小不够，进行填充
            if cell_patch.shape[0] < patch_size or cell_patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=cell_patch.dtype)
                h, w = cell_patch.shape[:2]
                padded_patch[:h, :w] = cell_patch
                cell_patch = padded_patch
            
            # 如果patch太大，进行裁剪
            elif cell_patch.shape[0] > patch_size or cell_patch.shape[1] > patch_size:
                cell_patch = cell_patch[:patch_size, :patch_size]
            
            return cell_patch
            
        except Exception as e:
            print(f"提取细胞patch失败: {e}")
            # 返回空白patch
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    
    def extract_dino_features(self, cell_patches):
        """使用DINOv3提取细胞patches的特征（高性能批量处理）"""
        if len(cell_patches) == 0:
            return np.zeros((0, self.dinov3_feature_dim))
        
        print(f"开始提取 {len(cell_patches)} 个patches的DINOv3特征...")
        print(f"使用批处理大小: {self.dino_batch_size} (大幅优化GPU利用率)")
        
        features = []
        batch_size = self.dino_batch_size  # 使用更大的批处理大小
        
        # 分批处理所有patches
        total_batches = (len(cell_patches) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(cell_patches), batch_size), desc="DINOv3特征提取", total=total_batches):
                batch_patches = cell_patches[i:i+batch_size]
                
                # DINOv3处理（优化内存和计算）
                try:
                    # 检查是否有自定义的预处理器
                    if hasattr(self, 'dino_processor_transform'):
                        # 使用自定义的torchvision预处理器
                        processed_tensors = []
                        for patch in batch_patches:
                            # 确保是0-255的像素值
                            if patch.max() <= 1.0:
                                patch = (patch * 255).astype(np.uint8)
                            
                            # 转换为PIL图像
                            from PIL import Image
                            pil_image = Image.fromarray(patch)
                            
                            # 应用预处理
                            tensor = self.dino_processor_transform(pil_image)
                            processed_tensors.append(tensor)
                        
                        # 堆叠成batch
                        batch_tensor = torch.stack(processed_tensors)
                        
                    else:
                        # 使用HuggingFace的处理器
                        processed_images = self._parallel_preprocess_images(batch_patches)
                        inputs = self.dino_processor(images=processed_images, return_tensors="pt")
                        batch_tensor = inputs['pixel_values']
                    
                    # 移动到设备，让autocast管理精度
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    # 不强制转换为FP16，让autocast自动管理
                    # if torch.cuda.is_available():
                    #     batch_tensor = batch_tensor.to(self.device, non_blocking=True, dtype=torch.float16)
                    # else:
                    #     batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    
                    # 提取特征（启用自动混合精度）
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):  # 更新的autocast语法
                            if hasattr(self, 'dino_processor_transform'):
                                # torch.hub加载的模型，直接调用
                                batch_features = self.dino_model(batch_tensor)
                                # 如果返回的是tuple，取第一个元素
                                if isinstance(batch_features, tuple):
                                    batch_features = batch_features[0]
                                # 如果是4D tensor (batch, channels, height, width)，取global average pooling
                                if len(batch_features.shape) == 4:
                                    batch_features = batch_features.mean(dim=[2, 3])  # Global average pooling
                                elif len(batch_features.shape) == 3:
                                    batch_features = batch_features.mean(dim=1)  # 平均token特征
                            else:
                                # HuggingFace模型
                                outputs = self.dino_model(pixel_values=batch_tensor)
                                batch_features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        if hasattr(self, 'dino_processor_transform'):
                            batch_features = self.dino_model(batch_tensor)
                            if isinstance(batch_features, tuple):
                                batch_features = batch_features[0]
                            if len(batch_features.shape) == 4:
                                batch_features = batch_features.mean(dim=[2, 3])
                            elif len(batch_features.shape) == 3:
                                batch_features = batch_features.mean(dim=1)
                        else:
                            outputs = self.dino_model(pixel_values=batch_tensor)
                            batch_features = outputs.last_hidden_state.mean(dim=1)
                    
                    # 转换为numpy并检查NaN/Inf值
                    batch_features_np = batch_features.cpu().numpy()
                    
                    # 检查并清理NaN/Inf值
                    if np.isnan(batch_features_np).any() or np.isinf(batch_features_np).any():
                        print(f"  警告: 批次 {i//batch_size} 检测到NaN/Inf值，进行清理")
                        batch_features_np = np.nan_to_num(batch_features_np, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    features.append(batch_features_np)
                    
                    # 实时资源监控
                    if i % (batch_size * 5) == 0:  # 每5个大批次显示资源使用
                        resource_info = self.monitor_resources()
                        print(f"  资源使用: {resource_info}")
                    
                    # 减少内存清理频率，提高效率
                    if i % (batch_size * 20) == 0:  # 每20个大批次清理一次
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"批次 {i//batch_size} DINOv3处理失败: {e}")
                    # 使用零向量替代
                    zero_features = np.zeros((len(batch_patches), self.dinov3_feature_dim))
                    features.append(zero_features)
        
        # 合并所有特征
        all_features = np.vstack(features) if features else np.zeros((0, self.dinov3_feature_dim))
        
        print(f"DINOv3特征提取完成: {all_features.shape}")
        return all_features
    
    def _parallel_preprocess_images(self, batch_patches):
        """并行预处理图像patches"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            processed_images = list(executor.map(self._preprocess_single_image, batch_patches))
        return processed_images
    
    def _preprocess_single_image(self, patch):
        """预处理单个图像patch"""
        if patch.max() <= 1.0:
            patch = (patch * 255).astype(np.uint8)
        return Image.fromarray(patch)
    
    
    
    
    
    def load_wsi_image(self, wsi_path):
        """加载WSI图像"""
        try:
            # 尝试使用openslide加载WSI
            try:
                import openslide
                wsi = openslide.OpenSlide(wsi_path)
                # 获取最高分辨率级别
                level = 0
                wsi_image = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
                wsi_image = np.array(wsi_image.convert('RGB'))
                print(f"  使用openslide加载WSI: {wsi_image.shape}")
                return wsi_image
            except ImportError:
                print("  openslide不可用，尝试其他方法...")
                
            # 尝试使用cv2加载
            import cv2
            wsi_image = cv2.imread(wsi_path)
            if wsi_image is not None:
                wsi_image = cv2.cvtColor(wsi_image, cv2.COLOR_BGR2RGB)
                print(f"  使用cv2加载WSI: {wsi_image.shape}")
                return wsi_image
            
            # 尝试使用PIL加载
            from PIL import Image
            wsi_image = Image.open(wsi_path).convert('RGB')
            wsi_image = np.array(wsi_image)
            print(f"  使用PIL加载WSI: {wsi_image.shape}")
            return wsi_image
            
        except Exception as e:
            print(f"  加载WSI失败: {e}")
            return None

    def process_sample_with_independent_pca(self, sample_id):
        """处理单个样本，独立训练PCA，提取完直接保存（每例独立PCA版本）"""
        print(f"\n=== 处理空转样本: {sample_id} ===")
        print("每例独立训练PCA模型，128维DINOv3特征")
        
        # 加载样本数据
        sample_data = self.load_sample_data(sample_id)
        cellvit_df = sample_data['cellvit_df']
        wsi_path = sample_data['wsi_path']
        
        # 提取细胞patches
        num_cells = len(cellvit_df)
        max_cells = num_cells  # 提取所有细胞，不限制数量
        cell_patches = []
        
        print(f"准备提取所有 {num_cells} 个细胞的特征...")
        
        # 先尝试加载WSI图像的小区域来测试
        print("尝试加载WSI图像...")
        try:
            import openslide
            wsi = openslide.OpenSlide(wsi_path)
            print(f"  WSI尺寸: {wsi.dimensions}")
            print(f"  WSI级别数: {wsi.level_count}")
            print(f"  WSI级别尺寸: {wsi.level_dimensions}")
            
            # 使用级别1分辨率，提供合适的细胞覆盖范围
            level = 1  # 使用级别1，~0.5μm/pixel，48x48像素覆盖24x24μm
            print(f"  使用级别 {level}，尺寸: {wsi.level_dimensions[level]}")
            
            # 从真实WSI中提取细胞patches（大批量并行处理）
            batch_size = self.cell_batch_size  # 使用优化后的批处理大小
            num_batches = (max_cells + batch_size - 1) // batch_size
            
            print(f"  将分 {num_batches} 批处理，每批 {batch_size} 个细胞 (大幅优化处理效率)")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, max_cells)
                batch_cells = end_idx - start_idx
                
                print(f"\n  处理批次 {batch_idx+1}/{num_batches}: 细胞 {start_idx}-{end_idx-1} ({batch_cells} 个)")
                
                # 并行提取patches以大幅提高CPU利用率
                print(f"\n  使用{self.num_workers}个并行进程提取patches...")
                batch_patches = self._extract_patches_parallel(
                    cellvit_df.iloc[start_idx:end_idx], wsi, level, start_idx
                )
                
                # 将本批次的patches添加到总列表
                cell_patches.extend(batch_patches)
                
                # 优化内存清理
                del batch_patches
                gc.collect()
                
                print(f"  批次 {batch_idx+1} 完成，累计提取 {len(cell_patches)} 个patches")
                
                # 实时GPU内存状态监控
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  当前GPU内存使用: {gpu_memory:.2f} GB")
            
            wsi.close()
            print(f"  成功从WSI提取 {len(cell_patches)} 个真实细胞patches")
            
        except Exception as e:
            print(f"  WSI加载失败: {e}")
            print("  无法从WSI中提取真实细胞图像，程序终止")
            raise RuntimeError(f"WSI图像加载失败，无法继续特征提取: {e}") from e
        
        # 提取DINOv3特征
        print("提取DINOv3特征...")
        dino_features = self.extract_dino_features(cell_patches)
        
        # 为当前样本独立训练PCA降维（不融合形态特征）
        print(f"为样本 {sample_id} 独立训练PCA降维...")
        final_features = self.apply_independent_pca(dino_features, sample_id)
        
        # 准备元数据
        metadata = {
            'sample_id': sample_id,
            'num_cells': len(cell_patches),
            'feature_dim': self.final_feature_dim,
            'dino_dim': self.final_dino_dim,
            'patch_size': self.cell_patch_size,
            'wsi_level': 1,  # 使用级别1
            'total_cells_processed': len(cell_patches),
            'independent_pca': True,
            'pca_trained_on_sample': sample_id
        }
        
        # 保存特征
        output_file = self.save_features(sample_id, final_features, metadata)
        
        print(f"\n=== 性能统计 ===")
        total_cells = len(cell_patches)
        print(f"✓ 总处理细胞数: {total_cells:,}")
        print(f"✓ DINOv3批处理大小: {self.dino_batch_size}")
        print(f"✓ 细胞批处理大小: {self.cell_batch_size}")
        print(f"✓ 并行工作者数: {self.num_workers}")
        print(f"✓ 最终特征维度: {self.final_feature_dim} (样本独立PCA)")
        print(f"✓ 特征文件: {output_file}")
        if torch.cuda.is_available():
            print(f"✓ 最终GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return {
            'sample_id': sample_id,
            'num_cells': len(cell_patches),
            'final_feature_dim': self.final_feature_dim,
            'output_file': output_file
        }
    
    def _extract_patches_parallel(self, cellvit_batch, wsi, level, start_idx):
        """使用多线程并行提取细胞patches以最大化CPU利用率"""
        batch_patches = []
        
        # 使用线程池而不是进程池，因为WSI对象不能序列化
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 创建任务列表
            tasks = []
            for i, (_, row) in enumerate(cellvit_batch.iterrows()):
                task = executor.submit(
                    self._extract_single_patch_threaded,
                    row['geometry'], 
                    wsi, 
                    level, 
                    self.cell_patch_size,
                    start_idx + i
                )
                tasks.append(task)
            
            # 收集结果
            for task in tqdm(tasks, desc=f"批次并行提取patches"):
                try:
                    patch = task.result(timeout=10)  # 10秒超时
                    batch_patches.append(patch)
                except Exception as e:
                    print(f"  并行提取失败，使用默认patch: {e}")
                    batch_patches.append(np.zeros((self.cell_patch_size, self.cell_patch_size, 3), dtype=np.uint8))
        
        return batch_patches
    
    def _extract_single_patch_threaded(self, geom_bytes, wsi, level, patch_size, cell_idx):
        """线程安全的单个细胞patch提取，使用黑色填充无效区域"""
        try:
            geom = wkb.loads(geom_bytes)
            centroid = geom.centroid
            center_x, center_y = centroid.x, centroid.y
            
            # 计算在指定级别下的坐标
            scale_factor = wsi.level_downsamples[level]
            half_size = patch_size // 2
            
            # 计算WSI边界
            wsi_width, wsi_height = wsi.level_dimensions[level]
            
            # 计算提取区域坐标
            x_start = int(center_x - half_size * scale_factor)
            y_start = int(center_y - half_size * scale_factor)
            x_end = x_start + int(patch_size * scale_factor)
            y_end = y_start + int(patch_size * scale_factor)
            
            # 创建黑色填充的patch
            cell_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            
            # 计算WSI内的有效区域
            valid_x_start = max(0, x_start)
            valid_y_start = max(0, y_start)
            valid_x_end = min(wsi_width * scale_factor, x_end)
            valid_y_end = min(wsi_height * scale_factor, y_end)
            
            # 如果有有效区域，提取该部分
            if valid_x_start < valid_x_end and valid_y_start < valid_y_end:
                # 计算在WSI中的实际提取尺寸
                region_width = int((valid_x_end - valid_x_start) / scale_factor)
                region_height = int((valid_y_end - valid_y_start) / scale_factor)
                
                if region_width > 0 and region_height > 0:
                    # 提取有效区域
                    region = wsi.read_region(
                        (valid_x_start, valid_y_start),
                        level,
                        (region_width, region_height)
                    )
                    
                    # 转换为RGB数组
                    region_array = np.array(region.convert('RGB'))
                    
                    # 计算在patch中的放置位置
                    patch_x_start = max(0, int((valid_x_start - x_start) / scale_factor))
                    patch_y_start = max(0, int((valid_y_start - y_start) / scale_factor))
                    patch_x_end = min(patch_size, patch_x_start + region_array.shape[1])
                    patch_y_end = min(patch_size, patch_y_start + region_array.shape[0])
                    
                    # 确保不超出边界，并将有效区域放入patch
                    if (patch_x_end > patch_x_start and patch_y_end > patch_y_start and 
                        region_array.shape[0] > 0 and region_array.shape[1] > 0):
                        
                        # 调整region_array大小以匹配目标区域
                        target_height = patch_y_end - patch_y_start
                        target_width = patch_x_end - patch_x_start
                        
                        if region_array.shape[:2] != (target_height, target_width):
                            region_array = cv2.resize(region_array, (target_width, target_height))
                        
                        cell_patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = region_array
            
            return cell_patch
            
        except Exception as e:
            if cell_idx < 5:  # 只显示前5个错误
                print(f"    细胞 {cell_idx} patch提取失败: {e}")
            # 返回纯黑色patch作为默认值
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    
    def apply_independent_pca(self, dino_features, sample_id):
        """为当前样本独立训练PCA降维（每例独立PCA版本）"""
        print(f"为样本 {sample_id} 独立训练PCA降维器...")
        
        # 检查并清理NaN值
        print(f"  检查数据质量...")
        nan_count = np.isnan(dino_features).sum()
        inf_count = np.isinf(dino_features).sum()
        
        if nan_count > 0:
            print(f"  警告: 发现 {nan_count} 个NaN值，将被替换为0")
            dino_features = np.nan_to_num(dino_features, nan=0.0)
        
        if inf_count > 0:
            print(f"  警告: 发现 {inf_count} 个Inf值，将被替换为有限值")
            dino_features = np.nan_to_num(dino_features, posinf=1.0, neginf=-1.0)
        
        # 检查是否还有有效的特征
        if np.all(dino_features == 0):
            print(f"  错误: 所有特征都为0，无法进行PCA")
            raise ValueError(f"样本 {sample_id} 的所有DINOv3特征都为0，可能是特征提取失败")
        
        # 动态调整PCA维度，不能超过样本数
        n_samples = dino_features.shape[0]
        n_features = dino_features.shape[1]
        max_components = min(n_samples, n_features)
        
        actual_dino_dim = min(self.final_dino_dim, max_components - 1)
        
        if actual_dino_dim < self.final_dino_dim:
            print(f"  警告: PCA维度从 {self.final_dino_dim} 调整为 {actual_dino_dim} (样本数限制)")
        
        if actual_dino_dim <= 0:
            print(f"  错误: 无法确定有效的PCA维度")
            raise ValueError(f"样本 {sample_id} 无法确定有效的PCA维度，样本数={n_samples}, 特征数={n_features}")
        
        # 为当前样本独立训练PCA
        from sklearn.decomposition import PCA
        try:
            sample_pca = PCA(n_components=actual_dino_dim, random_state=42)
            reduced_features = sample_pca.fit_transform(dino_features)
        except Exception as e:
            print(f"  PCA训练失败: {e}")
            print(f"  特征统计: min={dino_features.min():.6f}, max={dino_features.max():.6f}, mean={dino_features.mean():.6f}, std={dino_features.std():.6f}")
            raise ValueError(f"样本 {sample_id} PCA训练失败: {e}") from e
        
        # 保存当前样本的PCA模型
        sample_pca_path = os.path.join(self.output_dir, f"{sample_id}_dino_pca_model.pkl")
        with open(sample_pca_path, 'wb') as f:
            pickle.dump(sample_pca, f)
        
        # 计算方差解释比例
        explained_variance = sample_pca.explained_variance_ratio_.sum()
        explained_variance_each = sample_pca.explained_variance_ratio_
        
        print(f"样本 {sample_id} PCA训练完成:")
        print(f"  - 输入维度: {dino_features.shape[1]}")
        print(f"  - 输出维度: {actual_dino_dim}")
        print(f"  - 总解释方差比例: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        # 显示前10个主成分的方差解释比例
        print(f"  - 前10个主成分的方差解释比例:")
        for i in range(min(10, len(explained_variance_each))):
            print(f"    PC{i+1}: {explained_variance_each[i]:.4f} ({explained_variance_each[i]*100:.2f}%)")
        
        # 显示累积解释比例
        cumulative_variance = np.cumsum(explained_variance_each)
        print(f"  - 累积解释比例:")
        milestones = [10, 20, 50, 100, 128]
        for milestone in milestones:
            if milestone <= len(cumulative_variance):
                print(f"    前{milestone}个主成分: {cumulative_variance[milestone-1]:.4f} ({cumulative_variance[milestone-1]*100:.2f}%)")
        
        print(f"  - PCA模型保存: {sample_pca_path}")
        
        return reduced_features
    
    
    def save_features(self, sample_id, combined_features, metadata):
        """保存提取的特征"""
        output_file = os.path.join(self.output_dir, f"{sample_id}_combined_features.npz")
        
        np.savez_compressed(
            output_file,
            features=combined_features,
            metadata=metadata
        )
        
        print(f"特征已保存: {output_file}")
        return output_file



def get_all_hest_samples(hest_data_dir):
    """获取所有可用的HEST样本ID"""
    import glob
    import os
    
    # 查找所有cellvit分割文件
    cellvit_pattern = os.path.join(hest_data_dir, "cellvit_seg", "*_cellvit_seg.parquet")
    cellvit_files = glob.glob(cellvit_pattern)
    
    # 提取样本ID
    sample_ids = []
    for file_path in cellvit_files:
        filename = os.path.basename(file_path)
        sample_id = filename.replace('_cellvit_seg.parquet', '')
        
        # 检查对应的WSI文件是否存在
        wsi_path = os.path.join(hest_data_dir, "wsis", f"{sample_id}.tif")
        if os.path.exists(wsi_path):
            sample_ids.append(sample_id)
    
    sample_ids.sort()  # 按字母顺序排序
    return sample_ids


def get_progress_file(output_dir):
    """获取进度文件路径"""
    return os.path.join(output_dir, "extraction_progress.json")

def load_progress(output_dir):
    """加载处理进度"""
    progress_file = get_progress_file(output_dir)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"✓ 加载进度文件: {progress_file}")
            return progress
        except Exception as e:
            print(f"⚠️  加载进度文件失败: {e}")
            return {}
    return {}

def save_progress(output_dir, progress):
    """保存处理进度"""
    progress_file = get_progress_file(output_dir)
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        print(f"✓ 保存进度: {progress_file}")
    except Exception as e:
        print(f"⚠️  保存进度失败: {e}")

def is_sample_completed(output_dir, sample_id):
    """检查样本是否已完成处理"""
    output_file = os.path.join(output_dir, f"{sample_id}_combined_features.npz")
    pca_file = os.path.join(output_dir, f"{sample_id}_dino_pca_model.pkl")
    return os.path.exists(output_file) and os.path.exists(pca_file)

def main_independent_pca_extraction():
    """主函数：空转数据独立PCA特征提取（每例独立PCA）- 支持断点续传"""
    
    # 配置参数
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    output_dir = "/data/yujk/hovernet2feature/hest_spatial_features_independent_pca_dinov3"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有可用的HEST样本
    all_samples = get_all_hest_samples(hest_data_dir)
    print(f"\n发现 {len(all_samples)} 个可用样本: {all_samples}")
    
    # 加载处理进度
    progress = load_progress(output_dir)
    completed_samples = set(progress.get('completed_samples', []))
    failed_samples = set(progress.get('failed_samples', []))
    
    # 检查文件系统中已完成的样本
    file_completed_samples = set()
    for sample_id in all_samples:
        if is_sample_completed(output_dir, sample_id):
            file_completed_samples.add(sample_id)
    
    # 合并进度信息
    all_completed = completed_samples.union(file_completed_samples)
    
    # 确定需要处理的样本
    remaining_samples = [s for s in all_samples if s not in all_completed]
    
    print(f"\n=== 断点续传状态 ===")
    print(f"总样本数: {len(all_samples)}")
    print(f"已完成样本: {len(all_completed)} - {sorted(list(all_completed))}")
    print(f"失败样本: {len(failed_samples)} - {sorted(list(failed_samples))}")
    print(f"待处理样本: {len(remaining_samples)} - {sorted(remaining_samples)}")
    
    if not remaining_samples:
        print("✅ 所有样本已处理完成！")
        return
    
    # 询问是否从断点继续
    if all_completed:
        print(f"\n检测到 {len(all_completed)} 个已完成的样本")
        try:
            resume = input("是否从断点继续处理剩余样本？(y/n, 默认y): ").strip().lower()
            if resume in ['n', 'no']:
                print("用户选择重新开始处理")
                # 重置进度
                remaining_samples = all_samples
                progress = {'completed_samples': [], 'failed_samples': []}
                save_progress(output_dir, progress)
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return
    
    test_samples = remaining_samples  # 只处理剩余样本
    
    print("=== HEST空转数据独立PCA特征提取（DINOv3）- 断点续传版 ===")
    print(f"数据目录: {hest_data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"特征配置: 仅DINOv3，每例独立PCA至128维")
    print(f"使用级别1分辨率WSI (~0.5μm/pixel)")
    print(f"48×48像素patches，覆盖24×24μm物理区域")
    
    # 创建特征提取器（使用优化参数）
    try:
        # 自动检测最佳参数
        num_workers = min(mp.cpu_count(), 16)  # 限制最大16个线程
        
        # GPU内存检测
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"检测到GPU内存: {gpu_memory_gb:.1f} GB")
            
            # 根据GPU内存自动调整批处理大小
            if gpu_memory_gb >= 24:  # 24GB+
                dino_batch_size = 512
                cell_batch_size = 100000
            elif gpu_memory_gb >= 16:  # 16-24GB
                dino_batch_size = 384
                cell_batch_size = 80000
            elif gpu_memory_gb >= 12:  # 12-16GB
                dino_batch_size = 256
                cell_batch_size = 60000
            elif gpu_memory_gb >= 8:   # 8-12GB
                dino_batch_size = 128
                cell_batch_size = 40000
            else:  # <8GB
                dino_batch_size = 64
                cell_batch_size = 20000
        else:
            print("未检测到GPU，使用CPU模式")
            dino_batch_size = 32
            cell_batch_size = 10000
        
        print(f"自动优化配置:")
        print(f"  - CPU核心数: {mp.cpu_count()}, 使用线程数: {num_workers}")
        print(f"  - DINOv3批大小: {dino_batch_size}")
        print(f"  - 细胞批大小: {cell_batch_size:,}")
        
        # 设置 DINOv3 模型路径
        dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        
        extractor = HESTCellFeatureExtractor(
            hest_data_dir=hest_data_dir,
            output_dir=output_dir,
            dinov3_model_path=dinov3_model_path,  # 传入DINOv3模型路径
            bulk_pca_model_path=None,  # 不使用bulk PCA
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dinov3_feature_dim=1024,  # DINOv3-L 特征维度
            dino_batch_size=dino_batch_size,
            cell_batch_size=cell_batch_size,
            num_workers=num_workers
        )
        
        # 处理所有空转样本
        print(f"\n=== 开始处理 {len(test_samples)} 个空转样本 ===")
        print(f"准备处理样本: {', '.join(test_samples[:5])}{'...' if len(test_samples) > 5 else ''}")
        
        sample_results = []
        
        # 性能监控
        import time
        start_time = time.time()
        
        # 处理每个样本（使用独立PCA + 断点续传）
        for sample_idx, sample_id in enumerate(test_samples):
            print(f"\n样本处理进度: {sample_idx+1}/{len(test_samples)}")
            print(f"{'='*50}")
            print(f"正在处理空转样本: {sample_id}")
            print(f"{'='*50}")
            
            # 检查样本是否已完成
            if is_sample_completed(output_dir, sample_id):
                print(f"✅ 样本 {sample_id} 已完成，跳过")
                # 添加到结果中以便统计
                try:
                    output_file = os.path.join(output_dir, f"{sample_id}_combined_features.npz")
                    data = np.load(output_file)
                    num_cells = data['features'].shape[0]
                    final_feature_dim = data['features'].shape[1]
                    sample_results.append({
                        'sample_id': sample_id,
                        'num_cells': num_cells,
                        'final_feature_dim': final_feature_dim,
                        'output_file': output_file
                    })
                except Exception as e:
                    print(f"⚠️  读取已完成样本 {sample_id} 信息失败: {e}")
                continue
            
            # 异常处理：如果某个样本处理失败，记录并继续处理下一个
            try:
                sample_start_time = time.time()
                result = extractor.process_sample_with_independent_pca(sample_id)
                sample_end_time = time.time()
                sample_time = sample_end_time - sample_start_time
                
                sample_results.append(result)
                
                # 更新进度：添加到已完成列表
                current_progress = load_progress(output_dir)
                if 'completed_samples' not in current_progress:
                    current_progress['completed_samples'] = []
                if sample_id not in current_progress['completed_samples']:
                    current_progress['completed_samples'].append(sample_id)
                
                # 从失败列表中移除（如果存在）
                if 'failed_samples' in current_progress and sample_id in current_progress['failed_samples']:
                    current_progress['failed_samples'].remove(sample_id)
                
                save_progress(output_dir, current_progress)
                
                print(f"\n样本 {sample_id} 处理完成 (耗时: {sample_time:.1f}秒):")
                print(f"  - 处理细胞数: {result['num_cells']:,}")
                print(f"  - 最终特征维度: {result['final_feature_dim']}")
                print(f"  - 特征文件: {result['output_file']}")
                print(f"  - 处理速度: {result['num_cells']/sample_time:.0f} 细胞/秒")
                
                # 实时资源监控
                resource_info = extractor.monitor_resources()
                print(f"  - 当前资源使用: {resource_info}")
                
            except Exception as e:
                print(f"\n❌ 样本 {sample_id} 处理失败: {e}")
                print(f"   跳过该样本，继续处理下一个...")
                import traceback
                print(f"   错误详情: {traceback.format_exc()[:300]}...")
                
                # 更新进度：添加到失败列表
                current_progress = load_progress(output_dir)
                if 'failed_samples' not in current_progress:
                    current_progress['failed_samples'] = []
                if sample_id not in current_progress['failed_samples']:
                    current_progress['failed_samples'].append(sample_id)
                save_progress(output_dir, current_progress)
                
                continue  # 跳过失败的样本，继续处理下一个
        
        # 最终进度更新
        final_progress = load_progress(output_dir)
        if 'completed_samples' not in final_progress:
            final_progress['completed_samples'] = []
        
        # 确保所有成功处理的样本都在已完成列表中
        for result in sample_results:
            if result['sample_id'] not in final_progress['completed_samples']:
                final_progress['completed_samples'].append(result['sample_id'])
        
        save_progress(output_dir, final_progress)
        
        # 检查是否有成功处理的样本
        if not sample_results:
            print("\n❌ 所有样本处理失败，无法继续")
            return
        
        if len(sample_results) < len(test_samples):
            failed_samples = set(test_samples) - set([r['sample_id'] for r in sample_results])
            print(f"\n⚠️  以下样本处理失败: {failed_samples}")
        
        # 计算总处理时间和性能统计
        end_time = time.time()
        total_time = end_time - start_time
        total_cells = sum(result['num_cells'] for result in sample_results)
        
        # 批量处理性能报告
        print(f"\n{'='*80}")
        print("=== 空转数据独立PCA处理完成 - 性能报告 ===")
        print(f"{'='*80}")
        
        print(f"🏆 批量处理统计:")
        print(f"  - 成功处理样本数: {len(sample_results)}")
        print(f"  - 跳过/失败样本数: {len(test_samples) - len(sample_results)}")
        print(f"  - 成功处理的样本: {[r['sample_id'] for r in sample_results]}")
        
        print(f"⚡ 性能统计:")
        print(f"  - 总处理时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"  - 总处理细胞数: {total_cells:,}")
        print(f"  - 平均处理速度: {total_cells/total_time:.0f} 细胞/秒")
        print(f"  - 每个样本平均时间: {total_time/len(sample_results):.1f}秒")
        print(f"  - 每个样本平均细胞数: {total_cells//len(sample_results):,}")
        
        print(f"📊 保存统计:")
        print(f"  - 成功保存样本数: {len(sample_results)}/{len(test_samples)}")
        print(f"  - 保存成功率: {len(sample_results)/len(test_samples)*100:.1f}%")
        
        print(f"✅ 独立PCA配置:")
        print(f"  - 处理样本数: {len(sample_results)}/{len(test_samples)}")
        print(f"  - 每个细胞特征维度: {extractor.final_feature_dim} (样本独立PCA)")
        print(f"  - WSI分辨率: 级别1 (~0.5μm/pixel)")
        print(f"  - Patch大小: {extractor.cell_patch_size}×{extractor.cell_patch_size}像素 (24×24μm)")
        print(f"  - 图像来源: 仅真实WSI细胞图像")
        print(f"  - 输出目录: {output_dir}")
        print(f"  - 性能优化: DINOv3批大小{extractor.dino_batch_size}, 细胞批大小{extractor.cell_batch_size:,}, 多线程{extractor.num_workers}个")
        
        print(f"\n📄 样本详细信息:")
        for result in sample_results:
            print(f"  - {result['sample_id']}: {result['num_cells']:,}细胞, {result['final_feature_dim']}维特征")
        
        # 最终资源使用统计
        final_resource_info = extractor.monitor_resources()
        print(f"\n  - 最终资源使用: {final_resource_info}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 初始化sample_results以防止UnboundLocalError
        if 'sample_results' not in locals():
            sample_results = []
        if 'test_samples' not in locals():
            test_samples = []
        if 'extractor' not in locals():
            print("❗ 模型初始化失败，无法继续处理")
            return
            
        print(f"\n❗ 处理中断，当前状态:")
        print(f"  - 处理样本数: {len(sample_results)}/{len(test_samples)}")
        if len(sample_results) > 0:
            print(f"  - 每个细胞特征维度: {sample_results[0].get('final_feature_dim', 'N/A')}")
        print(f"  - WSI分辨率: 级别1 (~0.5μm/pixel)")
        print(f"  - Patch大小: 48×48像素 (24×24μm)")
        print(f"  - 图像来源: 仅真实WSI细胞图像")
        print(f"  - 输出目录: {output_dir}")
        
        if len(sample_results) > 0:
            print(f"\n📄 已完成样本详细信息:")
            for result in sample_results:
                print(f"  - {result['sample_id']}: {result['num_cells']:,}细胞, {result['final_feature_dim']}维特征")




if __name__ == "__main__":
    # 运行空转数据独立PCA特征提取（仅DINOv3特征）
    print("HEST细胞特征提取器 - 仅使用DINOv3特征（独立PCA）")
    print("特征配置: DINOv3 768维 -> PCA降维至128维")
    print("不包含形态特征，每个样本独立训练PCA")
    print()
    
    try:
        main_independent_pca_extraction()
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()