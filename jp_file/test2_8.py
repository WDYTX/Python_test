import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取图片
image_path = 'C:\\Users\liu\Desktop\output_2\\0002.png'  # 替换为你的图片路径
image = Image.open(image_path)

# 定义前景区域的像素坐标（这里假设是一个矩形，你需要根据实际情况提供正确的坐标）
foreground_region = (310, 340, 370, 370)  # (左上角x, 左上角y, 右下角x, 右下角y)

# 创建画布和子图
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# 在第一个子图上绘制原图，并在前景区域添加红色框
ax[0].imshow(image)
rect = patches.Rectangle((foreground_region[0], foreground_region[1]),
                         foreground_region[2] - foreground_region[0],
                         foreground_region[3] - foreground_region[1],
                         linewidth=2, edgecolor='red', facecolor='none')
ax[0].add_patch(rect)
ax[0].set_title('Original Image with Foreground Region')

# 在第二个子图上绘制放大的前景区域
foreground_patch = image.crop(foreground_region)
ax[1].imshow(foreground_patch)
ax[1].set_title('Enlarged Foreground Region')

# 设置子图之间的间隔
plt.subplots_adjust(wspace=0.5)

# 显示图形
plt.show()

