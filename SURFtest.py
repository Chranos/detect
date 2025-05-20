import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from shapely.geometry import Polygon

def calculate_gray_ratio(image, contour, center, inner_ratio=0.75):
    """
    计算白球的内圆和外环的灰度比
    
    参数:
    image: 灰度图像
    contour: 白球轮廓
    center: 白球中心坐标
    inner_ratio: 内圆面积占比，默认为0.75（即3/4）
    
    返回:
    inner_gray: 内圆平均灰度
    outer_gray: 外环平均灰度
    ratio: 内圆/外环 灰度比
    """
    # 创建掩码，只保留白球内部区域
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # 计算白球的等效半径（基于面积）
    area = cv2.contourArea(contour)
    equivalent_radius = np.sqrt(area / np.pi)
    
    # 计算内圆半径（使其面积为整体的inner_ratio）
    inner_radius = int(equivalent_radius * np.sqrt(inner_ratio))
    
    # 创建内圆掩码
    inner_mask = np.zeros_like(image)
    cv2.circle(inner_mask, center, inner_radius, 255, -1)
    
    # 创建外环掩码（白球区域减去内圆区域）
    outer_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
    
    # 应用掩码提取内圆和外环区域
    inner_region = cv2.bitwise_and(image, image, mask=inner_mask)
    outer_region = cv2.bitwise_and(image, image, mask=outer_mask)
    
    # 计算平均灰度
    inner_gray = np.mean(inner_region[inner_mask > 0])
    outer_gray = np.mean(outer_region[outer_mask > 0])
    
    # 计算灰度比（内圆/外环）
    gray_ratio = inner_gray / outer_gray if outer_gray > 0 else 0
    
    return inner_gray, outer_gray, gray_ratio

def extract_roi(image, points):
    """
    从图像中提取指定四边形区域
    
    参数:
    image: 原始图像
    points: 四边形的四个顶点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    返回:
    roi_image: 提取的区域图像（矩形）
    """
    # 将点列表转换为numpy数组
    points = np.array(points, dtype=np.float32)
    
    # 计算ROI的宽度和高度
    # 使用Polygon计算外接矩形
    poly = Polygon(points)
    min_x, min_y, max_x, max_y = poly.bounds
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    
    # 定义目标点（矩形）
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(points, dst_points)
    
    # 进行透视变换
    roi_image = cv2.warpPerspective(image, M, (width, height))
    
    return roi_image

def measure_circularity(image_path, layer_type='middle', roi_points=None):
    """
    使用Otsu阈值法测量图片中白色物体（假定为球体）的圆度，
    并在白球内部使用Otsu方法找到较黑的区域作为气泡
    同时检测白球是否发生桥接(外接圆超出图像边界)
    
    参数:
    image_path: 图像路径或直接传入图像
    layer_type: 层级类型，可选 'upper'(上层), 'middle'(中层), 'under'(下层)
    roi_points: 可选，检测区域的四个角点坐标
    """
    # 读取图像
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
    else:
        # 假设image_path是已加载的图像
        image = image_path.copy()
    
    # 如果提供了ROI点，截取指定区域
    if roi_points is not None:
        image = extract_roi(image, roi_points)
    
    # 获取图像尺寸
    image_height, image_width = image.shape[:2]
    
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
        if layer_type == 'under':
            return 0, image.copy(), (0, 0), 0, 0, 0, None, None, None, False, 0, 0, 0
        else:
            return 0, image.copy(), (0, 0), 0, 0, 0, None, None, None, False
    
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
    
    # 判断是否桥接 - 检查外接圆是否超出图像边界
    is_bridging = False
    
    # 计算外接圆是否超出图像边界
    left_edge = center_x - radius
    right_edge = center_x + radius
    top_edge = center_y - radius
    bottom_edge = center_y + radius
    
    if left_edge < 0 or right_edge >= image_width or top_edge < 0 or bottom_edge >= image_height:
        is_bridging = True
    
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
    
    # 添加桥接信息
    bridging_text = "yes" if is_bridging else "no"
    cv2.putText(result_image, f"bridging: {bridging_text}", 
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 如果是下层，计算内圆与外环的灰度比
    if layer_type == 'under':
        inner_gray, outer_gray, gray_ratio = calculate_gray_ratio(gray, largest_contour, center)
        
        # 在图像上添加灰度比信息
        cv2.putText(result_image, f"inner_gray: {inner_gray:.2f}", 
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_image, f"outer_gray: {outer_gray:.2f}", 
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_image, f"gray_ratio: {gray_ratio:.4f}", 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 可视化内外区域
        vis_mask = result_image.copy()
        # 计算白球的等效半径
        eq_radius = np.sqrt(area / np.pi)
        # 计算内圆半径（面积为总面积的3/4）
        inner_radius = int(eq_radius * np.sqrt(0.75))
        # 绘制内圆边界（青色）
        cv2.circle(vis_mask, center, inner_radius, (255, 255, 0), 2)
        # 将可视化掩码叠加到结果图像上
        result_image = cv2.addWeighted(result_image, 0.7, vis_mask, 0.3, 0)
        
        return circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging, inner_gray, outer_gray, gray_ratio
    else:
        return circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging

def process_image_with_rois(image_path, rois, layer_type='middle'):
    """
    处理图像中的多个ROI区域
    
    参数:
    image_path: 图像路径
    rois: ROI区域列表，每个元素是四个点的坐标
    layer_type: 层级类型
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
        
    # 在原图上标记ROI区域
    marked_image = original_image.copy()
    
    results = []
    
    # 处理每个ROI
    for i, roi_points in enumerate(rois):
        print(f"\n处理ROI #{i+1}:")
        
        # 在原图上绘制ROI区域
        points_array = np.array(roi_points, dtype=np.int32)
        cv2.polylines(marked_image, [points_array], True, (0, 255, 255), 2)
        cv2.putText(marked_image, f"ROI #{i+1}", 
                   (int(points_array[0][0]), int(points_array[0][1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 分析当前ROI
        try:
            if layer_type == 'under':
                circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging, inner_gray, outer_gray, gray_ratio = measure_circularity(original_image, layer_type, roi_points)
                result = {
                    'roi_id': i+1,
                    'circularity': circularity,
                    'radius': radius,
                    'area': area,
                    'bubble_count': len(bubble_data),
                    'is_bridging': bool(is_bridging),
                    'inner_gray': float(inner_gray),
                    'outer_gray': float(outer_gray),
                    'gray_ratio': float(gray_ratio)
                }
            else:
                circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging = measure_circularity(original_image, layer_type, roi_points)
                result = {
                    'roi_id': i+1,
                    'circularity': circularity,
                    'radius': radius,
                    'area': area,
                    'bubble_count': len(bubble_data),
                    'is_bridging': bool(is_bridging)
                }
            
            # 保存处理后的ROI图像
            roi_name = f"roi_{i+1}_{layer_type}"
            cv2.imwrite(f"{roi_name}_result.png", result_image)
            
            # 添加结果到列表
            results.append(result)
            
            print(f"ROI #{i+1} 分析结果:")
            print(f"  圆度: {circularity:.4f}")
            print(f"  半径: {radius}像素")
            print(f"  面积: {area:.2f}平方像素")
            print(f"  气泡数量: {len(bubble_data)}")
            print(f"  是否桥接: {'是' if is_bridging else '否'}")
            
            if layer_type == 'under':
                print(f"  内圆平均灰度: {inner_gray:.2f}")
                print(f"  外环平均灰度: {outer_gray:.2f}")
                print(f"  灰度比(内圆/外环): {gray_ratio:.4f}")
                
        except Exception as e:
            print(f"处理ROI #{i+1}时出错: {str(e)}")
            results.append({'roi_id': i+1, 'error': str(e)})
    
    # 保存标记了ROI的原始图像
    cv2.imwrite("marked_rois.png", marked_image)
    
    # 保存JSON结果
    with open(f"analysis_results_{layer_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return marked_image, results

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='白球图像分析工具')
    parser.add_argument('--image', type=str, default="tt6.jpg", help='要分析的图像路径')
    parser.add_argument('--layer', type=str, default='middle', choices=['upper', 'middle', 'under'], 
                        help='图像层级类型: upper(上层), middle(中层), under(下层)')
    parser.add_argument('--roi', type=str,default="rois.json", help='ROI区域JSON文件路径，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]的列表')
    parser.add_argument('--single_roi', action='store_true', help='是否使用单个ROI模式')
    
    # 解析命令行参数
    args = parser.parse_args()
    image_path = args.image
    layer_type = args.layer
    
    # 读取ROI文件
    if args.roi:
        try:
            with open(args.roi, 'r') as f:
                rois = json.load(f)
                
            if args.single_roi and len(rois) > 0:
                # 单一ROI模式
                print(f"使用单一ROI模式分析图像: {image_path}")
                if layer_type == 'under':
                    circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging, inner_gray, outer_gray, gray_ratio = measure_circularity(image_path, layer_type, rois[0])
                else:
                    circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging = measure_circularity(image_path, layer_type, rois[0])
                    
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
                title_text = f'检测结果 - 圆度: {circularity:.4f}, 气泡: {len(bubble_data)}'
                if layer_type == 'under':
                    title_text += f', 灰度比: {gray_ratio:.4f}'
                plt.title(title_text)
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'roi_analysis_{layer_type}.png', dpi=300)
                plt.show()
                
                # 保存最终结果
                cv2.imwrite(f'roi_result_{layer_type}.png', result_image)
                
                print(f"\nROI分析完成! 结果已保存")
            else:
                # 多ROI模式
                print(f"开始处理图像 {image_path} 中的 {len(rois)} 个ROI区域...")
                marked_image, results = process_image_with_rois(image_path, rois, layer_type)
                
                # 显示标记了ROI的原始图像
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
                plt.title(f'图像中的ROI区域 (共{len(rois)}个)')
                plt.axis('off')
                plt.show()
                
                print(f"\n所有ROI分析完成! 结果已保存为JSON文件")
                
        except Exception as e:
            print(f"读取ROI文件出错: {str(e)}")
            print("使用默认模式处理整个图像")
            # 使用标准处理模式
            run_standard_mode(args)
    else:
        # 没有提供ROI文件，使用标准处理模式
        run_standard_mode(args)

def run_standard_mode(args):
    """运行标准处理模式（处理整个图像）"""
    image_path = args.image
    layer_type = args.layer
    
    print(f"开始使用Otsu方法自动测量白球的圆度和检测气泡... (层级: {layer_type})")
    
    try:
        # 使用Otsu方法进行测量
        if layer_type == 'under':
            result = measure_circularity(image_path, layer_type)
            circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging, inner_gray, outer_gray, gray_ratio = result
        else:
            circularity, result_image, center, radius, area, perimeter, thresh, bubble_thresh, bubble_data, is_bridging = measure_circularity(image_path, layer_type)
        
        print(f"\n自动分析结果:")
        print(f"层级类型: {layer_type}")
        print(f"白球圆度: {circularity:.4f}")
        print(f"白球圆心位置: {center}")
        print(f"白球半径: {radius}像素")
        print(f"白球面积: {area:.2f}平方像素")
        print(f"白球周长: {perimeter:.2f}像素")
        print(f"检测到的气泡数量: {len(bubble_data)}")
        print(f"是否桥接: {'是' if is_bridging else '否'}")
        
        # 如果是下层，显示灰度比信息
        if layer_type == 'under':
            print(f"\n灰度比分析:")
            print(f"内圆平均灰度: {inner_gray:.2f}")
            print(f"外环平均灰度: {outer_gray:.2f}")
            print(f"灰度比(内圆/外环): {gray_ratio:.4f}")
        
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
        title_text = f'检测结果 - 圆度: {circularity:.4f}, 气泡: {len(bubble_data)}'
        if layer_type == 'under':
            title_text += f', 灰度比: {gray_ratio:.4f}'
        plt.title(title_text)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'white_ball_{layer_type}_analysis.png', dpi=300)
        plt.show()
        
        # 保存最终结果
        cv2.imwrite(f'white_ball_{layer_type}_final.png', result_image)
        
        print(f"\n白球圆度和气泡检测完成! 结果已保存为 white_ball_{layer_type}_final.png")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()