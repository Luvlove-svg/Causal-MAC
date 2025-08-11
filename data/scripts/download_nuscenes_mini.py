import os
import requests
from tqdm import tqdm


def download_file(url, save_path):
    """下载大文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


if __name__ == "__main__":
    # NuScenes mini 数据集
    nuscenes_url = "https://www.nuscenes.org/data/v1.0-mini.tgz"
    save_path = r"D:\我的大学\ccf\agent\Causal-MAC\data\nuscenes\v1.0-mini.tgz"

    print(f"开始下载 NuScenes-mini 数据集 (约8GB)...")
    download_file(nuscenes_url, save_path)
    print("下载完成!")