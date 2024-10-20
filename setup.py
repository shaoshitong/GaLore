# import csv

# # 定义输入和输出文件路径
# tsv_file_path = './datasets/GLUE/sst2/dev.tsv'
# csv_file_path = './datasets/GLUE/sst2/dev.csv'

# # 读取 TSV 文件并写入 CSV 文件
# with open(tsv_file_path, 'r', encoding='utf-8') as tsv_file:
#     tsv_reader = csv.reader(tsv_file, delimiter='\t')

#     with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
#         csv_writer = csv.writer(csv_file)

#         # 将内容从 TSV 写入 CSV
#         for row in tsv_reader:
#             csv_writer.writerow(row)

# print(f"转换完成, CSV 文件已保存到 {csv_file_path}。")

from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="galore-torch",
    version="1.0",
    description="GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection",
    url="https://github.com/jiaweizzhao/GaLore",
    author="Jiawei Zhao",
    author_email="jiawei@caltech.edu",
    license="Apache 2.0",
    packages=["galore_torch"],
    install_requires=required,
)