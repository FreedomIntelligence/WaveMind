# WaveMind数据处理文档

## 📂 文件树结构
```bash
/WaveMind_ROOT_PATH_
├── data
│   ├── ImageNetEEG
│   │   ├── eeg_signals_raw_with_mean_std.pth  # 原始数据需下载
│   │   ├── Image/                              # 原始图像需下载
│   │   └── process.py                          # 预处理脚本
│   ├── SEED
│   │   ├── Preprocessed_EEG/                   # 原始数据需下载
│   │   └── process.py                          # 预处理脚本
│   ├── THING-EEG
│   │   ├── Data/                               # 原始数据需下载
│   │   ├── data_config.json
│   │   ├── download.py
│   │   └── process.py                          # 预处理脚本
│   ├── Total
│   │   ├── CLIP_groundTruth/                   # 生成的CLIP特征
│   │   ├── data_label.h5                       # 生成的HDF5文件
│   │   └── dataset_weights.pth                 # 训练时自动生成
│   ├── TUAB
│   │   ├── edf/                                # 原始数据需下载
│   │   ├── process_refine/                     # 临时缓存目录（处理后自动删除）
│   │   └── process.py                          # 预处理脚本
│   ├── TUEV
│   │   ├── edf/                                # 原始数据需下载
│   │   └── process.py                          # 预处理脚本
│   └── create_dataset_pkl.py                   # CLIP groundtruth生成脚本
```

## 数据处理流程

### 1. 环境准备

```bash
# 首次运行必须设置环境变量
export WaveMind_ROOT_PATH_=/path/to/WaveMind
# 或通过 Setup_Env.sh 自动设置
bash Setup_Env.sh /path/to/WaveMind
```

### 2. 下载原始数据

参考各数据集的下载说明（见下方各数据集详细说明）

### 3. 运行预处理脚本

```bash
cd $WaveMind_ROOT_PATH_/data

# 处理各数据集 (按需运行)
python ImageNetEEG/process.py
python THING-EEG/process.py
python SEED/process.py
python TUAB/process.py
python TUEV/process.py
```

### 4. 生成CLIP Groundtruth

```bash
# 所有数据集处理完成后，生成CLIP特征
python create_dataset_pkl.py
```

## HDF5数据规范

### 1. 键命名规范

| 数据集 | 训练集键 | 测试集键 | 跨被试集键 |
|--------|---------|---------|-----------|
| ImageNetEEG | `ImageNetEEG_train` | `ImageNetEEG_test` | `ImageNetEEG_cross` |
| THING-EEG | `thingEEG_train` | `thingEEG_test` | `thingEEG_cross` |
| SEED | `SEED_train` | `SEED_test` | `SEED_cross` |
| TUAB | `TUAB_train` | `TUAB_test` | `TUAB_cross` |
| TUEV | `TUEV_train` | `TUEV_test` | `TUEV_cross` |

**注**: 不同数据集使用不同的命名风格（保持历史兼容性）。

### 2. 数据结构

每个样本为结构化数组，包含以下字段：

```python
dtype = [
    ('eeg_data', np.float32, (32, 512)),    # EEG信号: 32通道 × 512采样点
    ('text_feature', np.float16, (768,)),   # CLIP嵌入: 768维向量
    ('caption', 'S192'),                    # 文本描述: 固定192字节
    ('label', dtype, shape),                # 标签: 整数或浮点数
    ('image_path', 'S192')                  # 图像路径: 固定192字节（可选）
]
```

### 3. 验证数据完整性

```python
import h5py
import os

hdf5_path = os.path.join(os.environ['WaveMind_ROOT_PATH_'], 'data/Total/data_label.h5')

with h5py.File(hdf5_path, 'r') as f:
    print("HDF5数据集键:")
    for key in f.keys():
        print(f"  - {key}: {f[key].shape}")
```

预期输出：
```
HDF5数据集键:
  - ImageNetEEG_cross: (N,)
  - SEED_train: (N,)
  - SEED_test: (N,)
  - SEED_cross: (N,)
  - thingEEG_train: (N,)
  - thingEEG_test: (N,)
  - thingEEG_cross: (N,)
  - TUAB_train: (N,)
  - TUAB_test: (N,)
  - TUAB_cross: (N,)
  - TUEV_train: (N,)
  - TUEV_test: (N,)
  - TUEV_cross: (N,)
```

## CLIP Groundtruth文件

生成位置：`data/Total/CLIP_groundTruth/`

| 文件名 | 维度 | 说明 |
|--------|------|------|
| `ImageNetEEG.pkl` | - | 40个类别名称列表 |
| `thingEEG_closeset.npy` | (1573, 768) | 闭集对象的平均图像特征 |
| `thingEEG_closeset.pkl` | - | 1573个对象名称列表 |
| `thingEEG.npy` | (200, 768) | 零样本对象的平均图像特征 |
| `thingEEG.pkl` | - | 200个对象名称列表 |
| `SEED.npy` | (3, 768) | 3个情绪类别的文本特征 |
| `SEED.pkl` | - | ['negative', 'neutral', 'positive'] |
| `TUAB.npy` | (2, 768) | 2个类别的文本特征 |
| `TUAB.pkl` | - | ['abnormal', 'normal'] |
| `TUEV.npy` | (6, 768) | 6个事件类型的文本特征 |
| `TUEV.pkl` | - | ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG'] |

## 数据模态类型

WaveMind支持两种数据模态：

### Brain Cognition (图像-EEG对)
- **数据集**: ImageNetEEG, THING-EEG
- **CLIP特征来源**: CLIP-ViT图像编码器
- **特征类型**: 图像嵌入 (768维)
- **保存方法**: `Convert_and_Save.save_to_hdf5_new()`

### Brain State (文本-EEG对)
- **数据集**: SEED, TUAB, TUEV
- **CLIP特征来源**: CLIP-BERT文本编码器
- **特征类型**: 文本嵌入 (768维)
- **保存方法**: `Convert_and_Save.process_and_save()`

## 数据集详细说明

### THING-EEG
#### EEG data
1. 从 https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/main/Preprocessed_data_250Hz 下载EEG数据
2. 解压到 `WaveMind_ROOT_PATH_/data/THING-EEG/Data/Preprocessed_data_250Hz`

#### 配对图像数据
1. 从 https://osf.io/y63gw/files 下载 `training_images.zip` 和 `test_images.zip`
2. 解压到 `WaveMind_ROOT_PATH_/data/THING-EEG/Data/images_set/training_images` 和 `test_images`

### ImageNetEEG
参考 https://github.com/perceivelab/eeg_visual_classification

### SEED
参考 http://bcmi.sjtu.edu.cn/~seed/

### TUAB
从 https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml 下载

### TUEV
从 https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml 下载

## 数据统计

我们提供了 `ShowHDF5Statistic.ipynb` 文件用于展示每个数据集的统计信息。

## 处理脚本说明

所有 `process.py` 文件已优化为：
- ✅ 无注释代码冗余
- ✅ 统一的错误处理
- ✅ 清晰的文档注释
- ✅ 一致的HDF5保存逻辑
- ✅ 自动内存管理和清理

详见各数据集目录下的 `process.py` 文件。
