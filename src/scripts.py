from PIL import Image
import os


def resize_images(source_folder, target_folder):
    """
    函数功能：将源文件夹下的图片按照保持宽高比的原则调整尺寸，大致接近1024 * 1024，并保存到目标文件夹
    参数：
    source_folder (str)：存放原始图片的文件夹路径
    target_folder (str)：保存处理后图片的文件夹路径
    """
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹下所有文件和文件夹的名称列表
    file_list = os.listdir(source_folder)
    for file_name in file_list:
        source_file_path = os.path.join(source_folder, file_name)
        # 判断是否是文件且是常见的图片格式
        if os.path.isfile(source_file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # 打开图片
                image = Image.open(source_file_path)
                width, height = image.size
                max_size = 512
                # 计算调整比例，保持宽高比
                ratio = min(max_size / width, max_size / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                # 调整图片尺寸
                image = image.resize((new_width, new_height), Image.LANCZOS)
                target_file_path = os.path.join(target_folder, file_name)
                # 保存修改后的图片到目标文件夹
                image.save(target_file_path)
                print(f"{file_name} 图片已成功处理并保存到 {target_folder}")
            except Exception as e:
                print(f"处理 {file_name} 图片时出现错误: {e}")


if __name__ == "__main__":
    # 在这里指定存放原始图片的文件夹的实际路径
    source_folder_path = "/home/lyw/project/RF-Solver-Edit/all_pictures"
    # 在这里指定保存处理后图片的文件夹的实际路径
    target_folder_path = "/home/lyw/project/RF-Solver-Edit/all_pictures_resize"
    resize_images(source_folder_path, target_folder_path)