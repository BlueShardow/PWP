import asyncio
import cv2 as cv
import numpy as np
import math
import time

def resize_frame(frame):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    new_height = 300
    new_width = int(new_height * aspect_ratio)

    return cv.resize(frame, (new_width, new_height)), new_height, new_width, aspect_ratio

def process_frame(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.medianBlur(frame, 5)
    frame = cv.bilateralFilter(frame, 9, 75, 75)

    return frame

def get_perspective_transform(frame, roi_points, width, height):
    src_pts = np.float32(roi_points)
    dst_pts = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])

    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
    warped = cv.warpPerspective(frame, H, (width, height))

    return warped

def sobel_edges(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)

    edges = cv.magnitude(sobel_x, sobel_y)
    edges = cv.normalize(edges, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    _, binary_edges = cv.threshold(edges, 50, 255, cv.THRESH_BINARY)
    binary_edges = cv.dilate(binary_edges, None, iterations = 1) # do this multiple times, and next function
    binary_edges = cv.erode(binary_edges, None, iterations = 1)
    binary_edges = cv.medianBlur(binary_edges, 5)
    binary_edges = cv.bilateralFilter(binary_edges, 9, 75, 75)

    contours, _ = cv.findContours(binary_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    binary_edges_bgr = cv.cvtColor(binary_edges, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLinesP(binary_edges, rho = 1, theta = np.pi / 180, threshold = 100, minLineLength = 100, maxLineGap = 150)
    line_frame = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    contour_frame = frame.copy()
    cv.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    """
    # Draw bounding boxes on the original frame
    output_frame = frame.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """

    return contour_frame, binary_edges_bgr, line_frame, lines

def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot((x2 - x1), (y2 - y1))

def calculate_angle(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

    while angle < 0:
        angle += 180

    while angle > 180:
        angle -= 180

    return angle

def calculate_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    num = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
    den = np.hypot((y2 - y1), (x2 - x1))

    return num / den

def is_parallel_lines(line1, line2, toward_tolerance, away_tolerance, distance_threshold):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    angle1 = calculate_angle(line1)
    angle2 = calculate_angle(line2)

    angle_diff = abs(angle1 - angle2)

    if angle_diff > toward_tolerance and (180 - angle_diff) > away_tolerance:
        return False

    # Horizontal or vertical check
    is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
    is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

    # Check alignment and proximity
    if is_horizontal1 and is_horizontal2:
        vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)

        return vertical_distance < distance_threshold

    elif not is_horizontal1 and not is_horizontal2:
        horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)

        return horizontal_distance < distance_threshold

    return False

def merge_lines(lines, width, height = 200, min_distance = 85, merge_angle_tolerance = 20, vertical_leeway = 1.3, horizontal_leeway = 1.1):
    def weighted_average(p1, w1, p2, w2):
        # Apply exponential scaling to weights based on their lengths
        scaled_w1 = w1 ** 1.1
        scaled_w2 = w2 ** 1.1
        return (p1 * scaled_w1 + p2 * scaled_w2) / (scaled_w1 + scaled_w2)

    def sort_line_endpoints(line):
        x1, y1, x2, y2 = line

        if x1 < x2 and y1 < y2:
            return x1, y1, x2, y2
        
        elif x1 < x2 and y1 > y2:
            return x1, y2, x2, y1

        elif x1 > x2 and y1 < y2:
            return x2, y1, x1, y2
        
        elif x1 > x2 and y1 > y2:
            return x2, y2, x1, y1
        
        else:
            return x1, y1, x2, y2
        
    def adjust_towards_center(x1, y1, x2, y2, width):
        center_x = width // 2
        adjustment_factor = 0.1  
        slope = (y2 - y1) / (x2 - x1)

        if slope == 0:
            return x1, y1, x2, y2
        
        elif slope > 0 and (x1 < center_x or x2 < center_x):
            return x1, y1, x2, y2
        
        elif slope > 0 and (x1 > center_x or x2 > center_x):
            adjustment_factor = slope / 100

            x1 = int(x1 - adjustment_factor * (x1 - center_x))
            x2 = int(x2 - adjustment_factor * (x2 - center_x))

            return x1, y1, x2, y2


        elif slope < 0 and (x1 > center_x or x2 > center_x):
            return x1, y1, x2, y2
        
        elif slope < 0 and (x1 < center_x or x2 < center_x):
            adjustment_factor = slope / 100

            x1 = int(x1 - adjustment_factor * (x1 - center_x))
            x2 = int(x2 - adjustment_factor * (x2 - center_x))

            return x1, y1, x2, y2

        else:
            return x1, y1, x2, y2
        
    def fix_slope(line, width):
        x1, y1, x2, y2 = line

        if x1 < width // 2 and x2 < width // 2:
            return x2, y1, x1, y2
        
        else:
            return x1, y1, x2, y2

    def merge_once(lines):
        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = sort_line_endpoints(line1)
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = sort_line_endpoints(line2)

                    if is_parallel_lines(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
                        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
                        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

                        if is_horizontal1 and is_horizontal2:
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)

                            if vertical_distance > min_distance * horizontal_leeway or horizontal_distance > min_distance:
                                continue

                        elif not is_horizontal1 and not is_horizontal2:
                            vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
                            horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)

                            if vertical_distance > min_distance or horizontal_distance > min_distance * vertical_leeway:
                                continue

                        # Merge lines using weighted averages
                        l2_len = line_length(line2)

                        new_x1 = weighted_average(new_x1, line_weight, x3, l2_len)
                        new_y1 = weighted_average(new_y1, line_weight, y3, l2_len)
                        new_x2 = weighted_average(new_x2, line_weight, x4, l2_len)
                        new_y2 = weighted_average(new_y2, line_weight, y4, l2_len)
                        
                        line_weight += l2_len
                        used[j] = True
            
            merged_lines.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            #print("merge lines in function", merged_lines)
            used[i] = True

        #print("merged lines in 2", merged_lines)
        return merged_lines

    # Convert to a flattened list of (x1, y1, x2, y2)
    lines = [line[0] for line in lines]

    # Perform iterative merging until lines stabilize
    prev_lines = []

    while not np.array_equal(prev_lines, lines):
        prev_lines = lines.copy()
        lines = merge_once(lines)

    return lines

def sort_line_endpoints(line):
    x1, y1, x2, y2 = line

    if x1 < x2 and y1 < y2:
        return x1, y1, x2, y2
    
    elif x1 < x2 and y1 > y2:
        return x1, y2, x2, y1

    elif x1 > x2 and y1 < y2:
        return x2, y1, x1, y2
    
    elif x1 > x2 and y1 > y2:
        return x2, y2, x1, y1
    
    else:
        return x1, y1, x2, y2

def draw_midline_lines(warped_frame, blended_lines, width):
    mid_line = []

    if len(blended_lines) < 2:
        return warped_frame, mid_line

    if not isinstance(blended_lines, list):
        print(f"Error: blended_lines should be a list, but got {type(blended_lines)}")
        return
    
    for i in range(len(blended_lines)):
        for j in range(i + 1, len(blended_lines)):
            line1 = blended_lines[i]
            line2 = blended_lines[j]

            if isinstance(line1, tuple) and isinstance(line2, tuple):
                x1, y1, x2, y2 = line1
                x3, y3, x4, y4 = line2

                if is_parallel_lines(line1, line2, 25, 15, 9999):
                    x_mid1 = (x1 + x3) // 2
                    y_mid1 = (y1 + y3) // 2
                    x_mid2 = (x2 + x4) // 2
                    y_mid2 = (y2 + y4) // 2

                    sorted_x1, sorted_y1, sorted_x2, sorted_y2 = sort_line_endpoints((x_mid1, y_mid1, x_mid2, y_mid2))

                    if sorted_x1 == sorted_x2:
                        sorted_x2 = sorted_x1

                    cv.line(warped_frame, (sorted_x1, sorted_y1), (sorted_x2, sorted_y2), (255, 0, 0), 2)

            elif isinstance(line1, int) and isinstance(line2, int):
                if line1 + 3 < len(blended_lines) and line2 + 3 < len(blended_lines):
                    x1, y1, x2, y2 = blended_lines[line1:line1 + 4]
                    x3, y3, x4, y4 = blended_lines[line2:line2 + 4]

                    if is_parallel_lines((x1, y1, x2, y2), (x3, y3, x4, y4), 25, 15, 9999):
                        x_mid1 = (x1 + x3) // 2
                        y_mid1 = (y1 + y3) // 2
                        x_mid2 = (x2 + x4) // 2
                        y_mid2 = (y2 + y4) // 2

                        sorted_x1, sorted_y1, sorted_x2, sorted_y2 = sort_line_endpoints((x_mid1, y_mid1, x_mid2, y_mid2))

                        cv.line(warped_frame, (sorted_x1, sorted_y1), (sorted_x2, sorted_y2), (255, 0, 0), 2)

                        mid_line.append(sorted_x1, sorted_y1, sorted_x2, sorted_y2)
                    
                else:
                    print(f"Skipping invalid line indices: {line1}, {line2}")
            else:
                print(f"Skipping invalid line format: {line1} (Type: {type(line1)}), {line2} (Type: {type(line2)})")

    return warped_frame, mid_line

# Above is lane dection, below is logic for how to move ____________________________________________________________________________________________________________________________

def determine_direction(lines, width):
    """
    0 = stop
    1 = left
    2 = straight
    3 = right

    if its smth like 012 then that means left, then straight, then right
    """
    
    if len(lines) >= 2:
        lines = merge_lines(lines, width)

    if len(lines) >= 2:
        print("stop")
        return 0

    else:
        x1, y1, x2, y2 = lines[0]

        if x1 < (width // 2) - 20 and x2 < (width // 2) - 20: # left
            print("Left")
            return 1
        
        elif x1 > (width // 2) + 20 and x2 > (width // 2) + 20: # right
            print("Right")
            return 3
        
        elif (x1 > (width // 2) - 20 and x1 < (width // 2) + 20) and (x2 > (width // 2) - 20 and x2 < (width // 2) + 20): # straight
            print("Straight")
            return 2
        
        elif x1 < (width // 2) - 20 and x2 > (width // 2) + 20:
            print("Left, Straight, Right")
            return 123
        
        elif x1 > (width // 2) + 20 and x2 < (width // 2) - 20:
            print("Right, Straight, Left")
            return 321
        



def display_fps(frame, start_time):
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    return frame

def main(): #_____________________________________________________________________________________________________________________________________________________________________________
    cap = cv.VideoCapture(0)

    frame_skip = 0
    frame_by_frame_mode = False

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 3

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        merged_lines = []

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        """
        if frame_count % frame_skip != 0:
            continue
        """

        frame, height, width, ratio = resize_frame(frame)
        preprocessed_frame = process_frame(frame)

        #"""
        roi_points = [
            (0, height),  # bottom left
            (width, height),  # bottom right
            (width - 20, 100),  # top right
            (20, 100)  # top left
        ]
        #"""

        mask = np.zeros_like(preprocessed_frame)
        roi_corners = np.array(roi_points, dtype=np.int32)
        cv.fillPoly(mask, [roi_corners], 255)
        roi_frame = cv.bitwise_and(preprocessed_frame, mask)

        roi_points_np = np.array(roi_points, np.int32).reshape((-1, 1, 2))
        cv.polylines(frame, [roi_points_np], True, (0, 255, 0), 2)

        warped_frame = get_perspective_transform(preprocessed_frame, roi_points, width, height)
        contour_frame, binary_frame, line_frame, lines = sobel_edges(warped_frame)

        warped_width = warped_frame.shape[1]
        merge_line_frame = warped_frame.copy()

        if lines is not None:
            merged_lines = merge_lines(lines, warped_width)

            for line in merged_lines:
                x1, y1, x2, y2 = line
                cv.line(merge_line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            _, mid_lines = draw_midline_lines(merge_line_frame, merged_lines, warped_width)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(line_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #print("Lines:", len(lines))

        display_fps(frame, start_time)

        cv.imshow("Frame", frame)
        cv.imshow("Preprocessed Frame", preprocessed_frame)
        cv.imshow("ROI Frame", roi_frame)
        cv.imshow("Warped Frame", warped_frame)
        cv.imshow("Binary Frame", binary_frame)
        #cv.imshow("Contour Frame", contour_frame)
        cv.imshow("Line Frame", line_frame)
        cv.imshow("Merged Lines", merge_line_frame)

        print("Merged Lines:", len(merged_lines))

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('f'):
            frame_by_frame_mode = not frame_by_frame_mode

        if frame_by_frame_mode:
            while True:
                key = cv.waitKey(0) & 0xFF

                if key == ord('n'):
                    break

                elif key == ord('q'):
                    cap.release()
                    cv.destroyAllWindows()
                    return

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
