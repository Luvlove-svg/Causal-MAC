# 创建解压脚本 extract.py
import tarfile
import sys

def extract_tgz(file_path, output_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)
        print(f"成功解压到 {output_dir}")

if __name__ == "__main__":
    extract_tgz(
        r"D:\我的大学\ccf\agent\Causal-MAC\data\nuscenes\v1.0-mini.tgz",
        r"D:\我的大学\ccf\agent\Causal-MAC\data\nuscenes\v1.0-mini"
    )