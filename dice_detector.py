def detect_dice(frame):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define thresholds for white color in HSV
    lower_white = np.array([0, 0, 225])  # Adjust as necessary
    upper_white = np.array([180, 55, 255])
    # Threshold the image to isolate white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask_cleaned = mask.copy()
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cube_list = []
    # Filter and draw contours
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if 5000 > area > 1000:  # Minimum size threshold
            # Approximate the contour to determine circularity
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Check if the shape is roughly rectangular (dice shape)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 3)
                cube_list.append(cv2.boundingRect(contour))
                # print("Dice detected")

    if len(cube_list) == 0:
        return frame
    
    
    cube_box = list(cube_list[0])
    cube_box[0]-=10
    cube_box[1]-=10
    cube_box[2]+=30
    cube_box[3]+=30
    top_left = cube_box[:2]
    bottom_right = (top_left[0] + cube_box[2], top_left[1] + cube_box[3])
    
    try:
        ff = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
        cube = cv2.cvtColor(ff.copy(), cv2.COLOR_BGR2GRAY)
        trash , cube = cv2.threshold(cube, 180, 255, cv2.THRESH_BINARY)

    except:
        return frame

    blurred_t = cv2.GaussianBlur(cube,(3,3),cv2.BORDER_DEFAULT)

    circles = cv2.HoughCircles(
        blurred_t,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution
        minDist=8,  # Minimum distance between detected circles
        param1=60,  # Higher threshold for edge detection
        param2=10,  # Threshold for center detection
        minRadius=7,  # Minimum circle radius
        maxRadius=10   # Maximum circle radius
    )

    if circles is not None:      
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # cv2.circle(ff, (x, y), r, (0, 255, 0), 2)
            # cv2.circle(ff, (x, y), 2, (0, 0, 255), 3)
        
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
        # put text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Dice: "+str(len(circles[0, :])), (top_left[0], top_left[1] - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return frame