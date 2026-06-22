#!/usr/bin/python3
"""
A simple gesture recognition application that uses MediaPipe to detect 
hand landmarks from a webcam feed and control a turtle graphics object 
based on the average position of the detected hand landmarks. The 
application initializes the camera, sets up the hand landmarker, and
creates a turtle graphics window. It continuously reads frames from 
the camera, detects hands, calculates the average position of the 
hand landmarks, and moves the turtle accordingly. The application also
displays the processed video feed with hand landmarks and average 
position information.

Written by: David Straat
Written on: 2026-jun-22
"""

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
import mediapipe as mp
from turtle import Screen, Turtle
import cv2 as cv


def main(frame_width=640, frame_height=480, dead_zone=0.15):
    """
    A simple gesture recognition application that uses MediaPipe to 
    detect hand landmarks from a webcam feed and control a turtle 
    graphics object based on the average position of the detected 
    hand landmarks. The application initializes the camera, sets 
    up the hand landmarker, and creates a turtle graphics window. 
    It continuously reads frames from the camera, detects hands, 
    calculates the average position of the hand landmarks, and 
    moves the turtle accordingly. The application also displays 
    the processed video feed with hand landmarks and average 
    position information.

    Parameters:
    frame_width (int): The width of the video frame.
    frame_height (int): The height of the video frame.
    dead_zone (float): The size of the dead zone as a fraction of 
                       the frame dimensions. The turtle will not 
                       move if the average hand position is within 
                       this dead zone.
    """
    
    # Initialize the camera 
    camera = cv2.VideoCapture(0)

    # Initialize the hand landmarker
    baseOptions = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=baseOptions, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Initialize the turtle graphics
    screen = Screen()
    screen.title("Tutle")
    screen.setup(width=frame_width, height=frame_height)
    turtle = Turtle()


    # Define the dead zone boundaries
    dead_zone_x_min = int((0.5 - dead_zone) * frame_width)
    dead_zone_x_max = int((0.5 + dead_zone) * frame_width)
    dead_zone_y_min = int((0.5 - dead_zone) * frame_height)
    dead_zone_y_max = int((0.5 + dead_zone) * frame_height)

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
    try:
        # Start the main loop to read frames from the camera
        while True:
            # Initiliaze average_x and average_y to the center of the frame
            average_x = frame_width // 2
            average_y = frame_height // 2

            # Read a frame from the camera
            ret, frame = camera.read()

            # Check if the frame was read successfully
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize the frame and convert it to RGB for MediaPipe processing
            frame = cv2.resize(frame, (frame_width, frame_height))
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Create a MediaPipe Image object from the RGB frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect hands in the frame using the hand landmarker
            result = detector.detect(mp_image)

            # Initialize lists to store the x and y coordinates of the detected hand landmarks.
            # This will be used to calculate the average position of the hand landmarks.
            x_list = []
            y_list = []

            # If hand landmarks are detected, process them to calculate the average position and draw on the frame
            if result.hand_landmarks:
                # Iterate through each detected hand's landmarks
                for hand_landmarks in result.hand_landmarks:
                    # Extract the x and y coordinates of each landmark and add them to the respective lists
                    x_list.extend([lm.x for lm in hand_landmarks])
                    y_list.extend([lm.y for lm in hand_landmarks])
                    mp_drawing.draw_landmarks(frame, hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS)

                # Calculate the average x and y coordinates of the detected hand landmarks and convert 
                # them to pixel values based on the frame dimensions
                average_x = int(np.mean(x_list) * frame_width)  
                average_y = int(np.mean(y_list) * frame_height) 

                # Draw a circle at the average position of the hand landmarks and display the average coordinates on the frame
                cv2.circle(frame, (average_x, average_y), 10, (255,0,255), -1)
                
                # Draw a rectangle and display the average coordinates of the hand landmarks on the frame
                cv2.rectangle(frame, (average_x + 15, average_y+5), (average_x + 150, average_y - 20), (255,255,255), cv2.FILLED)
                cv2.putText(frame, f"Avg: ({average_x}, {average_y})", (average_x + 15, average_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

            # Based on average position of hands, determine the direction to move the turtle.
            # Right side of the frame
            if average_x > dead_zone_x_max:
                distance = average_x - dead_zone_x_max
                turtle.right(10 * (distance / (0.4*frame_width)))
                x_pos = "right"

            # Left side of the frame
            elif average_x < dead_zone_x_min:
                distance = dead_zone_x_min - average_x
                turtle.left(10 * (distance / (0.4*frame_width)))
                x_pos = "left"

            # Center of the frame
            else:
                x_pos = "center"

            # Bottom side of the frame
            if average_y > dead_zone_y_max:
                distance = average_y - dead_zone_y_max
                turtle.backward(10 * (distance / (0.4*frame_height)))
                y_pos = "bottom"

            # Top side of the frame
            elif average_y < dead_zone_y_min:
                distance = dead_zone_y_min - average_y
                turtle.forward(10 * (distance / (0.4*frame_height)))
                y_pos = "top"

            # Center of the frame
            else:
                y_pos = "center"

            # Draw the dead zone rectangle
            cv2.rectangle(frame, (dead_zone_x_min, dead_zone_y_min), (dead_zone_x_max, dead_zone_y_max), (0, 255, 0), 2)
            
            # Display the position acquired from the hand landmarks on the frame
            # Determind through the if-else ladders above
            cv2.putText(frame, f"Position: {x_pos}, {y_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Display the processed frame in a window titled 'Hand Detection'
            cv2.imshow('Hand Detection', frame)\
            
            # Press "Q" to exit the loop and close the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass


main()