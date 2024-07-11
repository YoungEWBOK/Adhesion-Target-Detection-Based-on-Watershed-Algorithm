import cv2
import numpy as np

def show_resized(name, img, scale=0.3):
    resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    cv2.imshow(name, resized)

def watershed_algorithm(image):
    if image is None:
        print("Error: Input image is None.")
        return

    src = image.copy()
    
    try:
        # 边缘保留滤波EPF 去噪
        blur = cv2.pyrMeanShiftFiltering(image, sp=21, sr=55)
        show_resized("blur", blur)

        # 转成灰度图像
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # 得到二值图像区间阈值
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        show_resized('thres image', binary)

        # 距离变换
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        dist_out = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
        show_resized('distance-Transform', dist_out * 100)
        ret, surface = cv2.threshold(dist_out, 0.5*dist_out.max(), 255, cv2.THRESH_BINARY)
        show_resized('surface', surface)
        sure_fg = np.uint8(surface)  # 转成8位整型
        show_resized('Sure foreground', sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)  # 连通区域
        print(f"Number of connected components: {ret}")
        markers = markers + 1  # 整个图+1，使背景不是0而是1值

        # 未知区域标记(不能确定是前景还是背景)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
        unknown = binary - sure_fg
        show_resized('unknown', unknown)

        # 未知区域标记为0
        markers[unknown == 255] = 0
        # 区域标记结果
        markers_show = np.uint8(markers)
        show_resized('markers', markers_show*100)

        # 分水岭算法分割
        markers = cv2.watershed(image, markers=markers)
        markers_8u = np.uint8(markers)

        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                  (255,0,255), (0,255,255), (255,128,0), (255,0,128),
                  (128,255,0), (128,0,255), (255,128,128), (128,255,255)]

        object_count = 0
        areas = []
        for i in range(2, np.max(markers) + 1):
            mask = cv2.inRange(markers_8u, i, i)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                area = cv2.contourArea(contours[0])
                areas.append(area)

        # 使用直方图分析面积分布
        hist, bin_edges = np.histogram(areas, bins=20)
        most_common_bin = np.argmax(hist)
        min_area = bin_edges[most_common_bin]
        max_area = bin_edges[most_common_bin + 1]

        # 设定面积范围
        standard_area = (min_area + max_area) / 2
        area_threshold_low = standard_area * 0.7
        area_threshold_high = standard_area * 1.3

        for i in range(2, np.max(markers) + 1):
            mask = cv2.inRange(markers_8u, i, i)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                area = cv2.contourArea(contours[0])
                if area_threshold_low <= area <= area_threshold_high:
                    object_count += 1
                elif area > area_threshold_high:
                    # 如果面积超过阈值，认为是粘连物体，进行进一步分割
                    num_objects = round(area / standard_area)
                    object_count += num_objects
                    print(f"Detected adhered object with area {area:.2f}, estimated {num_objects} objects")
                
                color = colors[(i-2)%len(colors)]
                cv2.drawContours(image, contours, -1, color, -1)
                
                try:
                    M = cv2.moments(contours[0])
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])  # 轮廓重心
                        cv2.drawMarker(image, (cx,cy), (0,0,255), 1, 10, 2)
                        cv2.drawMarker(src, (cx,cy), (0,0,255), 1, 10, 2)
                        # 在图像上标注面积
                        cv2.putText(image, f"{area:.0f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                except Exception as e:
                    print(f"Error processing contour for region {i}: {e}")

        cv2.putText(src, f"count={object_count}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(image, f"count={object_count}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        show_resized('regions', image)
        result = cv2.addWeighted(src, 0.6, image, 0.5, 0)  # 图像权重叠加
        show_resized('result', result)

    except Exception as e:
        print(f"An error occurred during image processing: {e}")

# 主程序
try:
    src = cv2.imread('C:/Users/25705/Desktop/54.jpg')
    if src is None:
        raise IOError("Could not read the image file.")
    
    show_resized('src', src)
    watershed_algorithm(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except cv2.error as e:
    print(f"OpenCV Error: {e}")
except IOError as e:
    print(f"I/O error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")