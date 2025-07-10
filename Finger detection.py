import cv2
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a named window and set its properties
cv2.namedWindow('Hand Finger Counter', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Finger Counter', 900, 700)


def draw_instruction_panel(frame):
    """Draw instruction panel on the frame"""
    panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    # Title
    cv2.putText(panel, "Finger Counter", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    # Instructions
    instructions = [
        "Instructions:",
        "1. Place your hand in the green box",
        "2. Keep fingers spread apart",
        "3. Keep palm facing the camera",
        "4. Avoid complex backgrounds",
        "",
        "Controls:",
        "ESC - Exit application",
        "SPACE - Toggle mask view",
        "s - Save current frame"
    ]

    y = 90
    for i, line in enumerate(instructions):
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 40 if i == 0 else 30

    # Finger count color key
    cv2.putText(panel, "Visualization:", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(panel, "Contour - Green", (30, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(panel, "Convex Hull - Blue", (30, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(panel, "Finger Joints - Red", (30, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(panel, "Centroid - Yellow", (30, 530),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Combine panel with frame
    frame_with_panel = np.hstack((frame, panel))
    return frame_with_panel


show_mask = False
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Create a copy for drawing results
    display_frame = frame.copy()

    # Define ROI (Region of Interest)
    roi_top, roi_bottom, roi_left, roi_right = 100, 400, 150, 450
    cv2.rectangle(display_frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
    cv2.putText(display_frame, "Place hand here", (roi_left, roi_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Extract ROI
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Skip processing if ROI is empty
    if roi.size == 0:
        display_frame = draw_instruction_panel(display_frame)
        cv2.imshow('Hand Finger Counter', display_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        continue

    # Convert to HSV for skin color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range (adjust these values for different skin tones)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (presumably the hand)
        max_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(max_contour)

        # Skip if contour is too small
        if contour_area < 2000:
            display_frame = draw_instruction_panel(display_frame)
            cv2.imshow('Hand Finger Counter', display_frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            continue

        # Draw contour on ROI
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

        # Find convex hull
        hull = cv2.convexHull(max_contour)
        cv2.drawContours(roi, [hull], -1, (255, 200, 0), 2)

        # Find convexity defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        defects = []
        finger_count = 0

        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(max_contour, hull_indices)

        # Calculate centroid
        M = cv2.moments(max_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(roi, (cx, cy), 8, (0, 255, 255), -1)  # Draw centroid

        # Process convexity defects to count fingers
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate distances
                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)

                # Skip if any distance is zero
                if a == 0 or b == 0 or c == 0:
                    continue

                # Calculate angle
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                angle_deg = angle * 180 / math.pi

                # Calculate depth (actual distance)
                actual_depth = d / 256.0

                # Conditions for valid finger joint
                if (angle_deg < 90 and
                        actual_depth > 20 and
                        far[1] < cy and  # Only consider points above centroid
                        b > 20 and c > 20):  # Minimum finger length

                    cv2.circle(roi, far, 8, (0, 0, 255), -1)
                    finger_count += 1

        # Calculate total fingers (each defect represents a space between fingers)
        total_fingers = finger_count + 1 if finger_count > 0 else 0

        # Cap at 5 fingers
        total_fingers = min(total_fingers, 5)

        # Display finger count
        cv2.putText(display_frame, f'Fingers: {total_fingers}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Add instruction panel
    display_frame = draw_instruction_panel(display_frame)

    # Show mask if requested
    if show_mask:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (300, 300))
        display_frame[10:310, 10:310] = mask_bgr
        cv2.putText(display_frame, "Skin Mask", (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Display frame
    cv2.imshow('Hand Finger Counter', display_frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == 32:  # SPACE to toggle mask
        show_mask = not show_mask
    elif key == ord('s'):  # Save current frame
        cv2.imwrite(f'hand_gesture_{frame_count}.jpg', display_frame)
        print(f"Frame saved as hand_gesture_{frame_count}.jpg")

# Release resources
cap.release()
cv2.destroyAllWindows()
