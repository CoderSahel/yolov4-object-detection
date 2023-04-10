import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNetFromDarknet('yolov4-tiny-obj.cfg', 'yolov4-tiny-obj_4000.weights')

# Set input size and output layers
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
input_size = (416, 416)
output_layers = net.getLayerNames()
output_layers = [output_layers[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video capture device
cap = cv2.VideoCapture("set.mp4")

# Loop through video frames
while True:
    # Capture frame from video device
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1/255, input_size, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Run YOLOv4 on frame
    outputs = net.forward(output_layers)
    
    # Postprocess detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
