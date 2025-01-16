from asyncio import subprocess
import os
import logging
from flask import Flask, jsonify, request, render_template, send_file, flash, redirect, url_for, Response
import cv2
import math
import numpy as np
from werkzeug.security import check_password_hash, generate_password_hash
import pickle
import signal
import atexit

# =======================================================================
# Overlay-related helper functions (copied from your given overlay code)
# =======================================================================
def line_length(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(line):
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    while angle < 0:
        angle += 180
    while angle > 180:
        angle -= 180
    return angle

def is_parallel(line1, line2, toward_tolerance, away_tolerance, distance_threshold):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    angle1 = calculate_angle(line1)
    angle2 = calculate_angle(line2)

    angle_diff = abs(angle1 - angle2)
    angle_diff = min(angle_diff, 180 - angle_diff)

    if angle_diff > toward_tolerance and (180 - angle_diff) > away_tolerance:
        return False

    # Horizontal or vertical check
    is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
    is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

    if is_horizontal1 and is_horizontal2:
        vertical_distance = abs((y1 + y2) / 2 - (y3 + y4) / 2)
        return vertical_distance < distance_threshold
    elif not is_horizontal1 and not is_horizontal2:
        horizontal_distance = abs((x1 + x2) / 2 - (x3 + x4) / 2)
        return horizontal_distance < distance_threshold

    return False

def merge_close_lines_recursive(lines, min_distance, merge_angle_tolerance, vertical_leeway=1.5, horizontal_leeway=0.5):
    def weighted_average(p1, w1, p2, w2):
        return (p1 * w1 + p2 * w2) / (w1 + w2)

    def merge_once(lines):
        merged_lines = []
        used = [False] * len(lines)

        for i, line1 in enumerate(lines):
            if used[i]:
                continue

            x1, y1, x2, y2 = line1
            angle1 = calculate_angle(line1)
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            line_weight = line_length(line1)

            for j, line2 in enumerate(lines):
                if i != j and not used[j]:
                    x3, y3, x4, y4 = line2
                    angle2 = calculate_angle(line2)

                    # Check parallelism
                    if is_parallel(line1, line2, merge_angle_tolerance, merge_angle_tolerance, min_distance):
                        is_horizontal1 = abs(y2 - y1) < abs(x2 - x1)
                        is_horizontal2 = abs(y4 - y3) < abs(x4 - x3)

                        # Apply orientation-based logic
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

                        # Merge lines
                        l2_len = line_length(line2)
                        new_x1 = weighted_average(new_x1, line_weight, x3, l2_len)
                        new_y1 = weighted_average(new_y1, line_weight, y3, l2_len)
                        new_x2 = weighted_average(new_x2, line_weight, x4, l2_len)
                        new_y2 = weighted_average(new_y2, line_weight, y4, l2_len)
                        line_weight += l2_len
                        used[j] = True

            merged_lines.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
            used[i] = True

        return merged_lines

    prev_lines = []
    while prev_lines != lines:
        prev_lines = lines
        lines = merge_once(lines)
    return lines

def visualize_angles(frame, lines, color=(255, 255, 255)):
    for line in lines:
        x1, y1, x2, y2 = line
        angle = calculate_angle(line)
        midpoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.putText(frame, f"{angle:.1f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def line_intersection_at_y(line, y):
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return x1
    m = (y2 - y1) / (x2 - x1)
    x = x1 + (y - y1) / m
    return x

def draw_center_line_for_parallel_pairs(frame, parallel_lines):
    used = [False] * len(parallel_lines)
    pairs = []
    for i in range(len(parallel_lines)):
        if used[i]:
            continue
        for j in range(i + 1, len(parallel_lines)):
            if not used[j]:
                pairs.append((parallel_lines[i], parallel_lines[j]))
                used[i] = True
                used[j] = True
                break

    for lineA, lineB in pairs:
        p1 = np.array((lineA[0], lineA[1], 1))
        p2 = np.array((lineA[2], lineA[3], 1))
        q1 = np.array((lineB[0], lineB[1], 1))
        q2 = np.array((lineB[2], lineB[3], 1))
        if p1[1] > p2[1]:
            p2, p1 = p1, p2
        if q1[1] > q2[1]:
            q2, q1 = q1, q2
        midpoints = [0, 0]
        for i, y in enumerate([0, 1000]):
            m1 = np.cross(np.cross(p1, p2), np.array([0, 1, y]))
            m2 = np.cross(np.cross(q1, q2), np.array([0, 1, y]))
            midpoints[i] = (m1 / m1[2]) + (m2 / m2[2])
        e1 = np.cross(np.cross(midpoints[0], midpoints[1]), np.cross(p1, q1))
        e2 = np.cross(np.cross(midpoints[0], midpoints[1]), np.cross(p2, q2))
        e1 /= e1[2]
        e2 /= e2[2]

        try:
            cv2.line(frame, (int(e1[0]), int(e1[1])), (int(e2[0]), int(e2[1])), (255, 0, 0), 3)

        except:
            pass

def detect_and_classify_lines(frame, max_line_gap=85, toward_tolerance=65, away_tolerance=45, merge_angle_tolerance=65,
                              distance_threshold=999999999, min_distance=250, min_line_length=195, min_overlap_ratio=0.8,
                              proximity_threshold=20):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([112, 70, 105])
    upper_blue = np.array([290, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((proximity_threshold, proximity_threshold), np.uint8)
    blue_mask_dilated = cv2.dilate(blue_mask, kernel, iterations=1)
    edges = cv2.Canny(frame, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=max_line_gap
    )

    parallel_blue_lines = []
    non_parallel_blue_lines = []
    non_blue_lines = []

    if lines is not None:
        filtered_lines = [line[0] for line in lines if line_length(line[0]) >= min_line_length]

        blue_lines = []
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            mask_line = np.zeros_like(blue_mask)
            cv2.line(mask_line, (x1, y1), (x2, y2), 255, 2)
            overlap = cv2.bitwise_and(blue_mask, mask_line)
            overlap_ratio = np.sum(overlap > 0) / np.sum(mask_line > 0) if np.sum(mask_line > 0) > 0 else 0
            proximity_overlap = cv2.bitwise_and(blue_mask_dilated, mask_line)
            proximity_ratio = np.sum(proximity_overlap > 0) / np.sum(mask_line > 0) if np.sum(mask_line > 0) > 0 else 0

            if overlap_ratio >= min_overlap_ratio or proximity_ratio >= min_overlap_ratio:
                blue_lines.append((x1, y1, x2, y2))
            else:
                non_blue_lines.append((x1, y1, x2, y2))

        blue_lines = merge_close_lines_recursive(blue_lines, min_distance, merge_angle_tolerance)

        used_in_parallel = set()
        for i in range(len(blue_lines)):
            found_parallel = False
            for j in range(i + 1, len(blue_lines)):
                if is_parallel(blue_lines[i], blue_lines[j], toward_tolerance, away_tolerance, distance_threshold):
                    parallel_blue_lines.append(blue_lines[i])
                    parallel_blue_lines.append(blue_lines[j])
                    found_parallel = True
                    used_in_parallel.add(i)
                    used_in_parallel.add(j)
            if not found_parallel and i not in used_in_parallel:
                non_parallel_blue_lines.append(blue_lines[i])

    # Visualize angles for all lines
    visualize_angles(frame, parallel_blue_lines, color=(0, 255, 0))  # Green for parallel lines
    visualize_angles(frame, non_parallel_blue_lines, color=(0, 255, 255))  # Yellow for non-parallel lines
    #IMPORTANT #visualize_angles(frame, non_blue_lines, color=(0, 0, 255))  # Red for non-blue lines

    # Draw parallel blue lines in green
    for line in parallel_blue_lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green

    # Draw non-parallel blue lines in yellow
    for line in non_parallel_blue_lines:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow

    # Draw non-blue lines in red
    #for line in non_blue_lines:
        #x1, y1, x2, y2 = line
        #cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red

    # Now draw the center line for parallel pairs
    draw_center_line_for_parallel_pairs(frame, parallel_blue_lines)
    return frame, blue_mask, blue_mask_dilated

# Flask setup
app = Flask(__name__)

# Logging setup
logging.basicConfig(filename='api_output.log', level=logging.DEBUG)
log = logging.getLogger("werkzeug")
log.disabled = True
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 10)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

users_file = 'users.pkl'
if not os.path.exists(users_file):
    # Create the file if it doesn't exist
    open(users_file, 'wb').close()

def save_to_pkl(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pkl(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        return []

user_logged_in = False
# Routes
@app.route('/', methods=['GET', 'POST'])
def login_register():
    global user_logged_in
    if request.method == 'POST':
        # Check if the request is from AJAX (JSON data) or a regular form submission
        if request.is_json:
            data = request.get_json()
            form_type = data.get('form_type')

            if form_type == 'login':
                username_or_email = data.get('username_or_email')
                password = data.get('password')
                users = load_from_pkl(users_file)
                user = next((u for u in users if u['username'] == username_or_email or u['email'] == username_or_email),
                            None)

                if user and check_password_hash(user['password'], password):
                    user_logged_in = True
                    return jsonify(success=True, redirect_url=url_for('home'))
                else:
                    return jsonify(success=False, message="Invalid username/email or password")

            elif form_type == 'register':
                name = data.get('name')
                username = data.get('username')
                email = data.get('email')
                password = data.get('password')
                users = load_from_pkl(users_file)

                if any(u['username'] == username for u in users):
                    return jsonify(success=False, message="Username already exists")
                elif any(u['email'] == email for u in users):
                    return jsonify(success=False, message="Email already exists")
                else:
                    hashed_password = generate_password_hash(password)
                    users.append({'name': name, 'username': username, 'email': email, 'password': hashed_password})
                    save_to_pkl(users, users_file)
                    return jsonify(success=True, message="Registration successful! Please log in.")

        else:
            form_type = request.form['form_type']
            if form_type == 'login':
                username_or_email = request.form['username_or_email']
                password = request.form['password']
                users = load_from_pkl(users_file)
                user = next((u for u in users if u['username'] == username_or_email or u['email'] == username_or_email),
                            None)

                if user and check_password_hash(user['password'], password):
                    flash('Login successful')
                    user_logged_in = True
                    return redirect('/home')
                else:
                    flash('Invalid username/email or password')
            elif form_type == 'register':
                name = request.form['name']
                username = request.form['username']
                email = request.form['email']
                password = request.form['password']
                users = load_from_pkl(users_file)

                if any(u['username'] == username for u in users):
                    flash('Username already exists')
                elif any(u['email'] == email for u in users):
                    flash('Email already exists')
                else:
                    hashed_password = generate_password_hash(password)
                    users.append({'name': name, 'username': username, 'email': email, 'password': hashed_password})
                    save_to_pkl(users, users_file)
                    flash('Registration successful. Please login.')
                    return redirect('/')

    return render_template('login.html')


@app.route('/home')
def home():
    if user_logged_in:
        return render_template('index.html')
    else:
        return redirect('/')


def generate_frames():
    """Generator for camera frames."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_overlay_frames():
    """Generator for overlay-processed frames."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, _, _ = detect_and_classify_lines(frame.copy())
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/overlay_feed')
def overlay_feed():
    """Video feed with overlay."""
    return Response(generate_overlay_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')

def video_feed():

    def generate():

        while True:

            success, frame = camera.read()  # Read a frame from the camera

            if not success:

                break

            _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame to JPEG

            yield (b'--frame\r\n'

                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n') # Send the frame in HTTP response
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """Serve log files dynamically."""
    log_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            return file.read(), 200, {'Content-Type': 'text/plain'}
    else:
        return jsonify({'error': 'Log file not found'}), 404


@app.route('/move', methods=['POST'])
def move():
    content = request.json
    command = content.get('command')

    if command:
        if command == 'stop':
            string_command = r"~/bcm2835-1.70/Motor_Driver_HAT_Code/Motor_Driver_HAT_Code/Raspberry\ Pi/c/main stop"
            os.system(string_command)

        else:
            string_command = r"~/bcm2835-1.70/Motor_Driver_HAT_Code/Motor_Driver_HAT_Code/Raspberry\ Pi/c/main " + str(command)
            subprocess.run([string_command], shell=True)
        logging.info(f"Received command: {command}")

        return jsonify({'success': True, 'command': command}), 200

    else:
        return jsonify({'error': 'Invalid command'}), 400


# Cleanup function to release camera resources
def cleanup_resources():
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        logging.info("Camera resources released.")
        logging.info("Application shutdown gracefully.")

    cv2.destroyAllWindows()


# Register cleanup with atexit and signal handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, lambda sig, frame: cleanup_resources())
signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_resources())


if __name__ == '__main__':
    try:
        while True:
            # Read a frame from the camera
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from camera.")
                break

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
                print("Exiting display loop.")
                break
            
            # Process the frame to generate blue masks
            _, blue_mask, blue_mask_dilated = detect_and_classify_lines(frame.copy())
            
            # Display the masks
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Dilated Blue Mask", blue_mask_dilated)

            app.run(port=5111, debug=True)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
                print("Exiting display loop.")
                break
        
        # Cleanup resources after exiting the loop
        cleanup_resources()
        cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down...")
        cleanup_resources()
        cv2.destroyAllWindows()

    cleanup_resources()
    cv2.destroyAllWindows()

cleanup_resources()
cv2.destroyAllWindows()
