import json

# 读取JSON文件内容
with open('PIE-Benchmark/mapping_file.json', 'r') as file:
    data = json.load(file)

# 遍历JSON数据中的每个键值对
for key, value in data.items():
    if'mask' in value:
        del value['mask']
#
# 将处理后的数据写回到原文件（你也可以选择写入到新文件，修改文件名即可）
with open('PIE-Benchmark/mapping_file_nomask.json', 'w') as file:
    json.dump(data, file, indent=4)