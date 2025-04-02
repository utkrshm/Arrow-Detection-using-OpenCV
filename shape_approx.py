import cv2
import numpy as np
import math

def get_angle(a, b, c):
    """Calculate the angle between three points"""
    try:
        ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    except Exception:
        ang = math.degrees(0)
    return ang + 360 if ang < 0 else ang


def process_contours(frame, contours):
    result = frame.copy()
    
    for contour in contours[0]:
        area = cv2.contourArea(contour)
        
        # Calculate the convex hull
        hull = cv2.convexHull(contour)
        
        # Get the minimum area rectangle that bounds the contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Calculate width and height of the rectangle
        width = max(rect[1])
        height = min(rect[1])
        
        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate convexity (ratio of contour area to hull area)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # Get the rotated rectangle
        x, y, w, h = cv2.boundingRect(contour)

        if 1 < aspect_ratio < 2.5 and 0.7 < convexity < 1.0:
            
            # Draw bounding rect for visualization
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # cv2.putText(result, f"Convexity: {convexity}", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.4)
        
            # Analyze EVERY contour for angle properties
            # Approximate the contour
            epsilon = 0.06 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Number of points in the approximated contour
            num_points = len(approx)
                    
            # Calculate angles for all contours
            max_angle = 0
            
            if num_points >= 3:  # Need at least 3 points to calculate angles
                # Get the points from the approximated contour
                points = [tuple(pt[0]) for pt in approx]
                
                # Calculate angles between consecutive triplets of points
                angles = []
                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i+1) % len(points)]
                    p3 = points[(i+2) % len(points)]
                    angle = get_angle(p1, p2, p3)
                    angles.append(angle)
                
                # Calculate sum and find maximum angle
                max_angle = max(angles)
                
                if max_angle < 110:
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    # cv2.putText(result, f"Angle sum: {angle_sum} Max Angle: {max_angle}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), thickness=2)
        
    return result


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Camera not detected at index 1... Using index 0")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Camera not found at index 0 either, please find your camera index before running program")
        exit()

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Frame not found")
        break
    
    cv2.imshow("Original", frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    edges = cv2.Canny(blur, 200, 240)
    # cv2.imshow("Edges", edges)
    
    closing_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((1, 1), np.uint8)
    
    dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_kernel)
    # cv2.imshow("Dilation", dilated_edges)
    
    contours = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_frame = cv2.drawContours(frame.copy(), contours[0], -1, (0, 0, 255), 1)    
    cv2.imshow("All Contours", contour_frame)
    
    res = process_contours(frame, contours)
    cv2.imshow("Arrow Detection", res)
    
    if cv2.waitKey(1) == ord("q"):
        break
    

cap.release()
cv2.destroyAllWindows()