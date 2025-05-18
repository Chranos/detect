import cv2
import numpy as np
import matplotlib.pyplot as plt

def measure_circularity(image_path):
    """
    使用Otsu阈值法测量图片中白色物体（假定为球体）的圆度，
    并在白球内部使用Otsu方法找到较黑的区域作为气泡
    
    参数:
    image_path: 图像路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Otsu阈值法自动找到最佳阈值
    threshold_value, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"白球Otsu自动确定的阈值: {threshold_value}")
    
    # 使用形态学操作改善分割结果
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 找到所有轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0, image.copy(), (0, 0), 0, 0, 0, None, None, None
    
    # 找到最大轮廓（假设白球是最大的对象）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 平滑轮廓以减少噪声影响
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 计算面积
    area = cv2.contourArea(largest_contour)
    
    # 计算周长
    perimeter = cv2.arcLength(largest_contour, True)
    
    # 计算圆度：4π * 面积 / (周长)²
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 计算最小外接圆
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(center_x), int(center_y))
    radius = int(radius)
    
    # ===== 在白球内部使用Otsu方法找到较黑的区域作为气泡 =====
    
    # 创建掩码，只保留白球内部区域
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # 将掩码应用到原始灰度图像，只保留白球内部
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # 创建一个新的二值图像，只显示白球内较黑的区域
    # 使用Otsu方法在白球内部再次进行阈值处理
    bubble_mask = np.zeros_like(gray)
    # 只在白球区域内应用Otsu阈值
    bubble_roi = masked_gray[masked_gray > 0]
    if len(bubble_roi) > 0:
        # 计算白球内部区域的Otsu阈值
        bubble_threshold, _ = cv2.threshold(bubble_roi.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"气泡Otsu自动确定的阈值: {bubble_threshold}")
        
        # 在整个白球区域内应用这个阈值，找到较黑的区域
        _, bubble_thresh = cv2.threshold(masked_gray, bubble_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        print("无法计算气泡阈值")
        bubble_thresh = np.zeros_like(gray)
    
    # 对气泡使用形态学操作，消除噪声
    small_kernel = np.ones((3, 3), np.uint8)
    bubble_thresh = cv2.morphologyEx(bubble_thresh, cv2.MORPH_OPEN, small_kernel)
    bubble_thresh = cv2.morphologyEx(bubble_thresh, cv2.MORPH_CLOSE, small_kernel)

    # 1. 创建黑色区域的掩码（反转bubble_thresh）
    black_region_mask = cv2.bitwise_not(bubble_thresh)
    cv2.imwrite('black_region_mask.png', black_region_mask)

    # 2. 找到黑色区域的轮廓
    black_contours, _ = cv2.findContours(black_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 找到面积最大的轮廓（假设是白球主体）
    main_contour = max(black_contours, key=cv2.contourArea)

    # 4. 创建一个掩码，只保留主体轮廓内部的区域
    main_mask = np.zeros_like(bubble_thresh)
    cv2.drawContours(main_mask, [main_contour], 0, 255, -1)
    cv2.imwrite('main_mask.png', main_mask)

    # 5. 将原始bubble_thresh与主体掩码进行与操作，只保留黑色区域内的白色部分
    cleaned_bubble_thresh = cv2.bitwise_and(bubble_thresh, bubble_thresh, mask=main_mask)
    cv2.imwrite('cleaned_bubble_thresh.png', cleaned_bubble_thresh)

    # 6. 直接在处理后的图像上查找气泡轮廓
    bubble_contours, _ = cv2.findContours(cleaned_bubble_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 处理气泡轮廓
    bubble_data = []
    
    for i, contour in enumerate(bubble_contours):
        # 平滑气泡轮廓
        b_epsilon = 0.001 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, b_epsilon, True)
        
        # 计算特征
        b_area = cv2.contourArea(contour)
        b_perimeter = cv2.arcLength(contour, True)
        b_circularity = 4 * np.pi * b_area / (b_perimeter * b_perimeter) if b_perimeter > 0 else 0
        
        # 计算最小外接圆
        (b_x, b_y), b_radius = cv2.minEnclosingCircle(contour)
        b_center = (int(b_x), int(b_y))
        b_radius = int(b_radius)
        
        bubble_data.append({
            'id': i + 1,
            'contour': contour,
            'center': b_center,
            'radius': b_radius,
            'area': b_area,
            'perimeter': b_perimeter,
            'circularity': b_circularity
        })
    
    # 可视化结果
    result_image = image.copy()
    # 绘制白球轮廓
    cv2.drawContours(result_image, [largest_contour], 0, (0, 255, 0), 2)
    # 绘制白球的最小外接圆
    cv2.circle(result_image, center, radius, (0, 0, 255), 2)
    
    # 绘制气泡区域
    for bubble in bubble_data:
        # 绘制气泡轮廓（蓝色）
        cv2.drawContours(result_image, [bubble['contour']], 0, (255, 0, 0), 2)
        # 绘制气泡的最小外接圆（黄色）
        cv2.circle(result_image, bubble['center'], bubble['radius'], (0, 255, 255), 2)
        # 在气泡中心添加编号
        cv2.putText(result_image, f"{bubble['id']}", 
                    bubble['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 在图像上添加文本信息
    cv2.putText(result_image, f"write_thresh: {threshold_value}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    if len(bubble_roi) > 0:
        cv2.putText(result_image, f"bubble_thresh: {bubble_threshold}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(result_image, f"circularity: {circularity:.4f}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(result_image, f"bubble_number: {len(bubble_data)}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data

def main():
    """主函数"""
    print("开始使用Otsu方法自动测量白球的圆度和检测气泡...")
    
    # 加载图像
    image_path = "tt2.jpg"  # 假设白球图像名为tt2.jpg
    
    try:
        # 使用Otsu方法进行测量
        circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data = measure_circularity(image_path)
        
        print(f"\n自动分析结果:")
        print(f"白球圆度: {circularity:.4f}")
        print(f"白球圆心位置: {center}")
        print(f"白球半径: {radius}像素")
        print(f"白球面积: {area:.2f}平方像素")
        print(f"白球周长: {perimeter:.2f}像素")
        print(f"检测到的气泡数量: {len(bubble_data)}")
        
        # 打印气泡信息
        for bubble in bubble_data:
            print(f"气泡 #{bubble['id']}:")
            print(f"  中心位置: {bubble['center']}")
            print(f"  半径: {bubble['radius']}像素")
            print(f"  面积: {bubble['area']:.2f}平方像素")
            print(f"  周长: {bubble['perimeter']:.2f}像素")
            print(f"  圆度: {bubble['circularity']:.4f}")
        
        # 显示结果
        plt.figure(figsize=(15, 10))
        
        # 显示原始图像
        plt.subplot(2, 2, 1)
        original = cv2.imread(image_path)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        
        # 显示Otsu阈值处理结果(白球)
        plt.subplot(2, 2, 2)
        plt.imshow(thresh, cmap='gray')
        plt.title('Otsu阈值处理结果(白球)')
        plt.axis('off')
        
        # 显示气泡阈值处理结果
        plt.subplot(2, 2, 3)
        plt.imshow(bubble_thresh, cmap='gray')
        plt.title('Otsu阈值处理结果(气泡)')
        plt.axis('off')
        
        # 显示最终分析结果
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'检测结果 - 白球圆度: {circularity:.4f}, 气泡数量: {len(bubble_data)}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('white_ball_bubble_analysis.png', dpi=300)
        plt.show()
        
        # 保存最终结果
        cv2.imwrite('white_ball_bubble_final.png', result_image)
        
        print("\n白球圆度和气泡检测完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()