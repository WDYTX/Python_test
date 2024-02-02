import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 读取原始图像
image_path = 'C:\\Users\\liu\\Desktop\\GT_2\\0272.png'  # 替换为实际的图像路径
original_image = Image.open(image_path)

# 创建一个子图
fig, ax = plt.subplots()

# 显示原始图像
ax.imshow(original_image)

# 定义感兴趣区域的坐标
x, y = 260, 330   # 感兴趣区域的左上角坐标
width, height = 20, 20  # 感兴趣区域的宽度和高度

# 创建红色线框
rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')

# 将线框添加到子图上
ax.add_patch(rect)
ax.axis('off')
# 保存新图1
new_image1_path = 'C:\\Users\\liu\\Desktop\\0272\GT.png'
plt.savefig(new_image1_path, bbox_inches='tight', pad_inches=0)

# 关闭绘图
plt.close()

# 读取新图1
new_image1 = Image.open(new_image1_path)

# 获取感兴趣区域
region_of_interest = original_image.crop((x, y, x + width, y + height))

# 放大感兴趣区域
enlarged_region = region_of_interest.resize((width * 2, height * 2), Image.ANTIALIAS)

# 保存新图2
# new_image2_path = 'new_image2.png'
# enlarged_region.save(new_image2_path)

# 显示新图1和新图2（可选）
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(new_image1)
plt.title('New Image 1')

plt.subplot(1, 2, 2)
plt.imshow(enlarged_region)
plt.title('New Image 2')

plt.show()
