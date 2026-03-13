import os
import shutil
import subprocess
import zipfile
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm
from scipy.ndimage import affine_transform

# ==================== 配置路径 ====================
DATA_ROOT = "./datasets/UKB-MRI"          # 包含所有eid文件夹的根目录
CSV_PATH = "./datasets/Brain_Age.csv"    # 含 eid, sex, age_2 的CSV
ATLAS_MNI = "./AAL3/AAL3.nii.gz"        # 您选择的图谱（MNI空间）
ATLAS_TXT = "./AAL3/AAL3.nii.txt" # 区域标签文件（id, name）
OUTPUT_CSV = "./datasets/brain_features.csv"
TMP_DIR = "./tmp/ukb_temp"
os.makedirs(TMP_DIR, exist_ok=True)

# 读取参与者信息
df_info = pd.read_csv(CSV_PATH).set_index('eid')
# 优先使用 age_2，若缺失则使用 age_0
df_info['age'] = df_info['age_2'].fillna(df_info['age_0'])
# 删除年龄或性别缺失的参与者
df_info = df_info.dropna(subset=['age', 'sex'])
print(f"有效参与者数量: {len(df_info)}")


# ==================== 辅助函数 ====================
def parse_aal3_labels(txt_path):
    """
    从 AAL3v1.nii.txt 解析区域ID和名称。
    返回: (region_ids, region_names, id_to_name)
    """
    ids = []
    names = []
    id_to_name = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    ids.append(idx)
                    names.append(name)
                    id_to_name[idx] = name
                except ValueError:
                    continue
    return ids, names, id_to_name

def get_files(tmp_dir):
    base = os.path.join(tmp_dir, "T1")
    return {
        'pve_0': os.path.join(base, "T1_fast", "T1_brain_pve_0.nii.gz"),
        'affine': os.path.join(base, "transforms", "T1_to_MNI_linear.mat"),
        'warp': os.path.join(base, "transforms", "T1_to_MNI_warp.nii.gz"),
        'first_seg': os.path.join(base, "T1_first", "T1_first_all_fast_firstseg.nii.gz"),
        # 'first_vols': os.path.join(base, "T1_first", "T1_first_all_fast_origvols.txt"),
        'brain_to_mni': os.path.join(base, "T1_brain_to_MNI.nii.gz"),
    }

def warp_atlas_to_native(files):
    """将MNI空间图谱变换到个体T1空间（仅仿射，若存在非线性可扩展）"""
    atlas_mni=ATLAS_MNI # 图谱图像路径（如 AAL3v1_1mm.nii.gz）
    pve_file=files['pve_0'] # 参考图像（灰质概率图），用于确定输出网格
    affine_mat=files['affine'] # 个体空间→MNI 空间的仿射矩阵（.mat 文件路径或数组）
    output_path= os.path.join(tmp_dir, "atlas_native.nii.gz") # 输出图像路径（nii.gz）
    
    # 加载图谱图像
    atlas_img = nib.load(atlas_mni)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine  # 图谱体素→MNI物理空间的变换

    # 加载参考图像（个体T1空间）
    ref_img = nib.load(pve_file)
    ref_data = ref_img.get_fdata()
    ref_affine = ref_img.affine       # 参考图像体素→个体物理空间的变换

    # 读取从个体物理空间→MNI物理空间的仿射矩阵
    if isinstance(affine_mat, str):
        mat = np.loadtxt(affine_mat)  # FSL的.mat文件是4x4矩阵
    else:
        mat = np.array(affine_mat)

    # 构造从个体体素坐标到图谱体素坐标的复合变换：
    # 个体体素 -> 个体物理 (ref_affine) -> MNI物理 (mat) -> 图谱体素 (inv(atlas_affine))
    combined = np.linalg.inv(atlas_affine) @ mat @ ref_affine

    # 输出图像的shape（与参考图像一致）
    output_shape = ref_data.shape

    # 使用最近邻插值（order=0）重采样，保持区域标签的整数值
    transformed = affine_transform(
        atlas_data,
        matrix=combined[:3, :3],
        offset=combined[:3, 3],
        output_shape=output_shape,
        order=0,
        mode='constant',
        cval=0
    )

    # 保存结果，使用参考图像的affine
    out_img = nib.Nifti1Image(transformed.astype(np.int16), ref_affine)
    nib.save(out_img, output_path)    
    return output_path

def compute_roi_volumes(pve_file, atlas_file):
    pve_img = nib.load(pve_file)
    pve_data = pve_img.get_fdata().astype(np.float32)
    vox_vol = np.prod(pve_img.header.get_zooms())  # mm3
    
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata().astype(np.int16)
    
    assert pve_data.shape == atlas_data.shape, "Shape mismatch"
    
    volumes = {}
    for rid in region_ids:
        mask = (atlas_data == rid)
        vol = np.sum(pve_data[mask]) * vox_vol
        volumes[rid] = vol
    return volumes

def parse_first_volumes(first_vols_file):
    """可选：从FIRST提取体积（如果图谱未覆盖某些区域）"""
    if not os.path.exists(first_vols_file):
        return {}
    vols = {}
    with open(first_vols_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                vols[parts[1]] = float(parts[2])  # 结构名称 -> 体积
    return vols

# ==================== 主循环 ====================
# 加载图谱标签
region_ids, region_names, id_to_name = parse_aal3_labels(ATLAS_TXT)
print(f"共加载 {len(region_ids)} 个区域")

all_features = []
missing=[]

for zip_path in tqdm(glob(DATA_ROOT + "/*.zip")):
    base = os.path.basename(zip_path)
    eid = int(base.split('_')[0])
    print(f"\nProcessing {eid}...")
    try:
        if eid not in df_info.index:
            missing.append(eid)
            raise FileNotFoundError(f"eid {eid} 不在CSV有效列表中，跳过")
        
        tmp_dir = os.path.join(TMP_DIR, str(eid))
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp_dir)

        files = get_files(tmp_dir)
        if not os.path.exists(files['pve_0']):
            raise FileNotFoundError("pve_0 missing")
    except Exception as e:
        print(f"  Skipping: {e}")
        continue
    
    # 1. 变换图谱到个体空间
    atlas_native = warp_atlas_to_native(files)
    
    # 2. 计算区域体积
    roi_vols = compute_roi_volumes(files['pve_0'], atlas_native)
    
    # 3. （可选）合并FIRST体积
    # first_vols = parse_first_volumes(files['first_vols'])
    
    # 4. 整合特征
    age_val = df_info.loc[eid, 'age']
    sex_val = df_info.loc[eid, 'sex']
    feat = {'eid': int(eid), 'age': age_val, 'sex': sex_val}

    feat.update({f"atlas_{rid}": roi_vols.get(rid, np.nan) for rid in region_ids})
    # feat.update({f"first_{k}": v for k, v in first_vols.items()})
    all_features.append(feat)

    shutil.rmtree(tmp_dir, ignore_errors=True)

# ==================== 保存结果 ====================
if all_features:
    feat_df = pd.DataFrame(all_features).set_index('eid')
    feat_df.to_csv(OUTPUT_CSV)
    print(f"Saved {len(feat_df)} participants to {OUTPUT_CSV}")
else:
    print("No data processed.")
missing_path = os.path.join(os.path.dirname(OUTPUT_CSV), "missing_eids.txt")
with open(missing_path, 'w') as f:
    for eid in missing:
        f.write(f"{eid}\n")