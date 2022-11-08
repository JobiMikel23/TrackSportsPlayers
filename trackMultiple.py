from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint

csrtTracker = ['CSRT']
##Built in object trackers in OpenCv, there are 8 different ones which can be used.#

##Return the tracker objects, given the name of the tracker class. Later used in the multitracker

##Multitracker class, in OpenCV allows for multiple object tracking

def trackerName(selectedTracker): ##this function originally created to be able to select from multiple trackers from
    ##multi tracker, however only using CSRT tracker for now
    # make a tracker using the tracker names

    if selectedTracker == csrtTracker[0]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('This is not a valid tracker')
        print('The trackers which can be used are as follows')
        for x in csrtTracker:
            print(x)

    return tracker

if __name__ == '__main__':

  print("CSRT is the selected tracker type \n")

  for x in csrtTracker:
      print(x)

  print("\nTo get started, select a player by clicking and dragging a box over the player\n"
        "Then press the ESC key\n"
        "You can now press the s key to start,\nor another other key to select a new player, the box will disappear")
##selected tracker
  trackerType = "CSRT"

#Load football video
## for the tracker we need a video frame, and a location to track
##Video capture used to load the footage
cap = cv2.VideoCapture('Videos_Images/FootballFootage.avi')
#Get the first frame
ret, frame = cap.read()


##Draw red circles in each corner, first two integers are the coordinates,next size of circle and colour RGB
# cv2.circle(frame, (1, 90), 20, (0, 0, 255), -2)  #draw circle on upper left
# cv2.circle(frame, (1640, 100), 20, (0, 0, 255), -2)#upper right of pitch
# cv2.circle(frame, (1, 1020), 20, (0, 0, 255), -2)#bottom left of pitch
# cv2.circle(frame, (1920, 1020), 20, (0, 0, 255), -2)#bottom right of pitch

##array for video
vid_coords =  np.array([
        [1,90],
        [1640,100],
        [1,1020],
        [1920,1020]
    ])

birds_eye_coords = np.array([
        [280, 20],
        [470, 20],
        [340,330],
        [460, 330]
    ])
##Load the birds eye view football pitch
# img = cv2.imread('footballpitch.jpg',0)


# m = cv2.getPerspectiveTransform(np.float32(vid_coords),np.float32(birds_eye_coords))
#
# result = cv2.warpPerspective(img,m, (492,348))
# cv2.circle(img, (280, 20), 5, (0, 0, 255), -1)  #draw circle on upper left
# cv2.circle(img, (470, 20), 5, (0, 0, 255), -2)#upper right of pitch
# cv2.circle(img, (340, 330), 5, (0, 0, 255), -2)#bottom left of pitch
# cv2.circle(img, (460, 330), 5, (0, 0, 255), -2)#bottom right of pitch
# cv2.imshow('Pitch',result)
# cv2.imshow('P', img)
#cv2.imshow("asdas",result)
# def arrayTransform(self,vid_array,birdsEye_array):
#     assert vid_array.shape==(4,2)
#     assert birdsEye_array.shape(4,2)
#     self.M=cv2.getPerspectiveTransform(np.float32(vid_array),np.float32(birdsEye_array))
#     self.invM=cv2.getPerspectiveTransform(np.float32(vid_array),np.float32(birdsEye_array))



##pts1 = np.float32([[1, 90],[1640, 100],[1, 1020],[1920, 1020]])
##pts2 = np.float32([[0, 0],[400, 0],[0, 600],[400, 600]])
## matrix = cv2.getPerspectiveTransform(pts1,pts2)
##result = cv2.warpPerspective(img,matrix, (500,600))


#exit if cant read the footage

if not ret:
    print('Cannot load video')
    sys.exit(1)

##Use OpenCV function which is called selectROI. This allows us to select boxes to track objects
## Select boxes
selectedBoxes = []
randomColors = []

#The selectROI function does not work for multiple objects, so the function must be put in a loop to select all objects

while True:
    # Select the player with the boxes
    cv2.namedWindow('PlayerTrack', cv2.WINDOW_NORMAL)
    currentBox = cv2.selectROI('PlayerTrack', frame)
    selectedBoxes.append(currentBox)
    randomColors.append(((255,255 ,255) )) ##Allows for each box to be the colour white

    print("\nPress s to start tracking")
    print("Or if you want to select another player press another key\n"
          "And select and drag over the next player you want to track")
    k = cv2.waitKey(0) & 0xFF
    if (k == 115):  ##When s is selected for next object
        break

print('Selected bounding boxes {}'.format(selectedBoxes))


# Make a multitracker object
multiTracker = cv2.MultiTracker_create()

# initialize the multitracker
for currentBox in selectedBoxes:
    multiTracker.add(trackerName("CSRT"), frame, currentBox)

    # Process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    #Attempt to transform

    # Update the location of the objects in the next frames
    ret, boxes = multiTracker.update(frame)

    # Use enumerate to actually draw the selected objects
    for i, updatedBox in enumerate(boxes):
        p1 = (int(updatedBox[0]), int(updatedBox[1]))
        p2 = (int(updatedBox[0] + updatedBox[2]), int(updatedBox[1] + updatedBox[3]))
        cv2.rectangle(frame, p1, p2, randomColors[i], 2, 1)

    m = cv2.getPerspectiveTransform(np.float32(vid_coords), np.float32(birds_eye_coords))

    # result = cv2.warpPerspective(img, m, (492, 348))
    # Display the window

    cv2.imshow('PlayerTrack', frame)




    # Use the ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break