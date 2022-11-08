# TrackSportsPlayers
Research Method 
Aims and Objectives
The aim of this project was to produce a system that automatically detects players with any type of ball sport being played using captured sport footage. Secondly, I wanted my work to produce information on the players such as player movement patterns, which can be visualized in a bird’s eye view of a football pitch to ultimately help coaches analyze and improve their players.
From my research it was clear that I would be approaching this project from a computer vision standpoint. Computer vision in simple terms is a field of study which develops techniques to help computers “see” and understand the content of digital images such as photographs and videos [17]. Following this I decided the most suitable method for my project would be to use python and the OpenCV open source computer vision library [18]. It is said that OpenCV is designed for computational efficiency with a strong focus on real-time applications. One of the goals of OpenCV is to provide a simple-to-use computer vision infrastructure, that helps people build sophisticated computer vision applications [18]. I deemed this perfect for what I was trying to achieve in my project. 
Another library which I decided to use was Numpy. Numpy is a community-developed open-source library that allows for numerical computing in python. One of the features of numpy is a multidimensional Python array object along with array-aware functions that operate on it [19]. The Numpy array is a data structure that efficiently stores and accesses multidimensional arrays allowing a wide variety of numerical computation [19]. From my research into similar projects that used Python and OpenCV, I discovered that Numpy was used throughout and was essential. 


 ![1](/uploads/199a0fe1ba6d773354e668952306ab7d/1.png)

Figure 1 of OpenCV and Numpy being imported

The first objective of my work was to find camera footage of sport from a fixed angle. This is because using multi-camera and multi-angle footage, would make tracking the players very complicated, because of different camera techniques used such as panning, zooming, and different camera angles. Therefore, it would require a lot of time for me to achieve. Due to time constraints I decided to use camera footage from a fixed angle for my work. Once I found the appropriate footage, I needed to solve how to load the footage using OpenCV and python. 
OpenCV has a built-in function to capture and display footage, called cv2.VideoCapture(). To display videos using this function you must capture, read, and finally display the video. Firstly, to capture the video footage, I created a VideoCapture object with the argument being the destination and filename of the footage.

![2](/uploads/11ad212d9777fd14110381265eed2ffa/2.png)
 
Figure 2 using the cv2.VideoCapture function with videos folder and name of file 
 Next to read the footage I used cap.read() this returns a Boolean value, if the frame is read correctly it will be true. I assigned img to cap.read. 
 
![3](/uploads/3515218574f1748b1f5842f6d7a85555/3.png)

Figure 3 using cap.read()

Lastly I used the cv2.imshow() function to display the footage, with the first argument ‘Playertrack’ and second argument being img to display the variable img which is the captured footage. Additionally I used the cv2.waitkey() function which takes an argument in milliseconds, in this I used 1 millisecond so the program displays the frame, and continues to refresh and read the frame from cap.read(). Furthermore I used the cv2.namedWindow() function with cv2.WINDOW_NORMAL constant as this allows for the window which displays the footage, to be resizable and also adds tracking bars to the windows. 

![4](/uploads/1b1c373aea579120fe0fbdd6b4a31cc1/4.png)
  
Figure 4 using cv2.imshow, cv2.waitKey and cv2.namedWindow functions

![5](/uploads/2b623b0f3d84802e7c5e802fa3a19a92/5.png)
 
Figure 5 displaying the title of the window ‘PlayerTrack’ from using imshow.

![6](/uploads/b991f29004304ac7c8cdee7c133eb954/6.png)
 
Figure 6 showing footage being loaded using python and OpenCV

Once the footage had been loaded, the next task was to track the players. The first method I used to try do this was with the continuously adaptive mean-shift (CAMShift) algorithm[20]. Mean-Shift is a kernel-based tracking method which uses density-based appearance models to represent targets[20]. However when attempting to track players using CAMshift, it produced undesirable results as the tracking algorithm was attempting to track all the players at the same time, rather than individual tracking boxes. 

![7](/uploads/d4716a5e1d06623c0ff927d456f1f43e/7.png)
 
Figure 7
 
![8](/uploads/b721bae086400e8eee3e9c050211bbba/8.png)

Figure 7 and 8 showing the tracking algorithm working incorrectly and attempting to track all the players at the same time.
Following more research, I attempted to use a multi-tracker method, using select and bounding boxes. With this particular method the objects are selected rather than detected. OpenCV has a built-in Multi tracker class, which provides an implementation of multi-object tracking. In addition, OpenCV has a function called SelectROI, SelectROI allows you to select an object with bounding boxes. The SelectROI function does not work for multiple objects, therefore the function must be put in a loop to select all the player objects.
 
![9](/uploads/32da986fe72ff0689d93f2f1b1a1948a/9.png)

Figure 9 using a loop so selectROI function can select multiple player objects

Next using the multitracker class, I created and initialized a multitracker object, then I updated the location of the player objects in the next frames using multiTracker.update. Finally, I use the enumerate method to actually draw the boxes around the selected players. 
 
![10](/uploads/0fbc2c1e2b9cfecb086bbe2c9cadb6f8/10.png)

Figure 10 creating and initializing multitracker

![11](/uploads/131e886f4582da9cf454be7d288b2145/11.png)
 
Figure 11 updating the location of the players, and drawing the boxes using enumerate
The results from using this method were positive and allowed me to select and track multiple players,  by clicking and dragging over the desired player.
 
![12](/uploads/0f4414cf7c2d052a65ad525ef1eeadc8/12.png)

Figure 12 selecting the player by clicking and dragging bounding box over player.

![13](/uploads/484284f68148cec3e6969e72c706bf84/13.png)
 
Figure 13 bounding box round player, and player being tracked.

Even though tracking players was a success, there were still some drawbacks with this method for what I was trying to achieve. Using multitracker and bounding boxes meant that I had to manually select the players I wanted to track; however, I wanted my program to be able to automatically detect and track the players. Furthermore, the players being tracked would not be differentiated between each other, one of my aims was to be able to track players depending on what their kit color was. I wanted to do this so that it would be easier for the user to see what team is being tracked. However, the biggest issue with this form of tracking, was that to translate the players position into a birds-eye view of a football pitch proved to be very difficult. I was not able to produce this feature using the multi-tracker class. This led me to research more and find another technique using the yolov3 real-time object detection algorithm. 
What is YOLOv3? 
You Only Look Once, Version 3(YOLOv3) uses a Convolutional Neural Network (CNN) for doing object detection [16]. YOLOv3 uses score regions of objects, based on their similarities to predefined classes. With these regions they’re noted as positive detections when the scoring is high with which class they closely identify with [16]. For example, YOLOv3 can differentiate between two different types of animals, depending on which regions have a high score when compared to the classes of animals in YOLOv3. YOLOv3 does this by separating an image into a grid. These grid cells predict several boundary boxes around objects that score highly with the predefined classes. Then each boundary box has a confidence score of how accurate it deems the prediction to be and detects one object per bounding box [16].
YOLOv3 provides three files to be used for tracking; a text file with the previously mentioned predefined classes and their names, called coco. Names. As well as a configuration file and weights file, there are different variations of this file, in my work I decided to use yolo 320 configuration, and yolo 320 weights files. 
To extract information from the coco.names file, I created a variable called Coco_Names_File and assigned the file path and coco.names file to it. Next, I created a list named List_Of_Classes to store all of the class names in. Following this I opened the text file using with open and extracted the information from it based on new lines.  

![14](/uploads/2e58ac2bc0254a9feb8a434d4d24ecfd/14.png)
 
Figure 14 preview of some of the predefined classes in coco.names

![15](/uploads/8ddbc09003709ad28449ad93cf781fdf/15.png)
 
Figure 15 extracting the information from coco.names based on new lines 
Once this information had been extracted, I imported the Yolo configuration files and the Yolo weights file with their file paths, which I then used to create the CNN. The CNN requires both of these files.
 
![16](/uploads/a530d41e5a07f9532b1f13f8ed43532a/16.png)

Figure 16 importing the yolo configurations and weights file, and also creating the CNN using these same files.
Now I had successfully loaded the required files, extracted the information I needed, and created the CNN. My next task was to input the football footage to the network. 



It is not possible to just input the footage into the CNN from cv2.Videocapture and cap.read, as the CNN only accepts a particular type of format. The accepted format is known as blob, so I had to convert the captured footage into blob format. To do this I created a variable named blob which contained a function cv2.dnn.blobFromImage, this function converts the img variable into blob format. This blob function has several parameters. The first parameter is image which is used for the input image [21], in this case the footage which is assigned as img. Next is the scale factor which can be used to scale the image by some factor [21] I set the scalefactor to 1/255. Size is the third parameter, which is the spatial size the CNN expects. Since the yolo configuration file I am using is yolo-320. I created a variable named Width_Height with 320 assigned to it, this variable was used for the size parameter. Following this the mean average is required for a mean subtraction file, which is the mean of a 3-tuple of RGB values. For my work I set these as default 0,0,0 as there was no use for this function. The following parameter is swapRB which is set as the default value of 1 as it is not needed. Finally, the crop parameter needed to be set, this parameter is a Boolean to decide whether the image supplied needs to be cropped [21]. I set this parameter equal to false as there is no need to crop the footage. Once I had supplied the correct parameters for the cv2.dnn.blob.fromImage function, I set blob as in input for the CNN using net.setInput(blob). 
 
![17](/uploads/b26bf979c6027563e06d62fc587ff64b/17.png)

Figure 17, Width_Height variable used in size parameter set to 320.

![18](/uploads/e71873e8b751caf02c945828722fa9d7/18.png)

 
Figure 18 using cv2.dnn.blobFromImage function with required parameters, and inputting blob into the network
After successfully converting the footage to blob format for the CNN, I needed to find the objects that I wanted to detect. However, to do this I needed to obtain and look at the output layers of the CNN. CNN’s architecture has several layers, mainly convolutional layers and three output layers. The layers which are required for this scenario are the three output layers.

![19](/uploads/402d483cb6767e86432990c85d6e1242/19.jpg)
 
Figure 19 Convolutional Neural Network as seen in [16]. 
As shown in figure 19, the last three layers are the three output layers. These output layers are all different. Therefore, there will be three different values coming from the CNN. To acquire the output of these output layers, the names of all the layers including the convolutional layers must be retrieved first. I achieve this by using the function net.getLayerNames() [23]. To extract only the output layers from all the layer names, I used the net.getUnconnectedOutLayers () function which returns the index of the output layers[23]. Following this I used Layers[i[0] – 1 and for I in net.getUnonncectedOutLayers to iterate through the output layers, and acquire the first element with I -1 which will obtain the name of this output layer. Once the output names have been retrieved, they’re then sent as a forward pass to the network which will find the output of the output layers. I achieved this by using net.forward(Output_Layers).

![20](/uploads/5e7d6e7f4b6fbcae0b3d66d444736bd7/20.png)
 
Figure 20, obtaining the layernames and the output of the output layers
The output of the output layers(outputs)can now be used for finding objects(players) in the video footage. I created a detectObjects function for this purpose, using parameters outputs and the footage captured named img. To start the height, width and channel of the footage is retrieved using img.shape. Following this I created three lists, the first list Bounding_Box which will be used to store the corner points of the bounding_boxes. ClassIds to store the id of the class with the highest confidence. Finally, the confidence_values list to store the confidence of the determined class.

![21](/uploads/b69b6031a45e2c09aca589a1dae64227/21.png)
 
Figure 21, beginning of detect objects function
After this I implemented two for loops to iterate through the three outputs and also the bounding boxes. The boxes in this case are named detections. Next from the detections(bounding_Boxes) I removed the first 5 elements as they’re not relevant, and this will mean the class index with the highest confidence value can be found. This was done with detection [5:]. The index of the max value is found using np.argmax, from this the values of the max value index is saved as the confidence. 

![22](/uploads/28c5db04c479484eb3b9485c52a36dfa/22.png)
 
Figure 22, two for loops to iterate through boxes and detections
Following this I used an if statement, if the confidence level is greater than the confidence threshold (in this case defines as 0.5 or 50%) then this will be saved as a positive detection. Within this if statement the values of w,h,x,y are obtained, this is done with width being stored in element [2] and height being stored in element [3]. These values are in float therefore, I multiplied by the width and height variables retrieved earlier on using img.shape. This returns the pixel value, and because of this they’re set as integers and not floating points. Now I have retrieved the height and width, the next step was to retrieve the x and y values. The x and y values are located at element 0 and element 1 respectively, and are multiplied by the width and height of img.shape. However, because X and Y are center points, I divided the w value by 2 and I did the same with the h value. Finally, I subtracted this from the width and the height. Now I have obtained w, h, x and y. I then appended them to the bounding box list, appended the max values index to the classids, and I appended the confidence to the confidence value list as a float.
 
![23](/uploads/f5a5188ad28d68c76c4c32e68f317e23/23.png)

Figure 23 If statement
With this, I managed to successfully draw the boxes on the players being tracked, however occasionally two detections would occur even though there is only one player object. To solve this problem, I used Non max suppression (NMS). NMS selects the best bounding box for an object and rejects the other bounding boxes. This is done with an objectiveness score [24] and the overlap of the bounding boxes. NMS selects the box with the highest objectiveness score, then compares the overlap of this high score box with the other boxes. NMS then deletes the box which is less than 50% and moves onto the next highest objectiveness score [24].
I implemented NMS using the OpenCV function cv2.dnn.NMSBoxes, with parameters bounding box, confidence value, confidence threshold and nmsThreshold. Further up in my code I defined nmsThreshold as 0. This function will send the bounding boxes in and returns what boxes to keep by giving the NMS.

![24](/uploads/e6626b4543c6f518dc0374769ddafe36/24.png)
 
Figure 24 cv2.dnn.NMSBoxes function for NMS 
The next task was to use NMS with drawing the bounding boxes. I achieved this using a for loop to iterate over the NMS. Within this for loop I obtain the x,y,w and h values using rect[0], rect[1],rect[2] and rect[3] respectively to achieve these values. After this I implemented the cv2.rectangle function to draw the boxes, within this function I included the x,y values and the corner values using x + width and y + height. I set the color of the boxes to be white using the RGB value 255,255,255 as well as setting the thickness of the boxes to the value of 2. Once I successfully managed to draw the boxes onto the players, I wanted to have labels for the players to be shown. I achieved this by using the cv2.putText function. Within this function I obtained the class id index using class ids[i], then I used this index to obtain the name from the list of classes I created earlier. I required for the label to be displayed in uppercase and achieved this using. upper (). Finally, I wanted the position of the text to be above the player, and so I used x, y-10, and set the font to italic using cv2.FONT_ITALIC. The scale was set to 0.8 and the color was set to white with an RGB value of (255,255,255), and the text thickness was set to the value of 2.
 
![25](/uploads/910e78ed18fd79a7dc8c363efc324d39/25.png)

Figure 25, for loop for drawing bounding boxes using nms

![26](/uploads/036231711bd793c68e4f942e6cb0c1a9/26.png)
 
Figure 26, Output of above code. Boxes drawn and labels
This produced a result where the players were being tracked and labels were being displayed on them. However, I wanted the players to be identified by the color of the kit they were wearing. For now, my work tracked all the players as if they were on the same team. To achieve this, I would need to create a custom object detector with Yolov3.
The first step to creating a custom object detector with yolov3, is to create a dataset of images of the players. I constructed this dataset by obtaining images of the players from the captured sport footage. With this footage I then screenshotted the footage once all of the players were in the frame. With this screenshot, I cropped the relevant players from the image depending on the color of their kit. I repeated this process several times at different timestamps of the footage, allowing me to acquire images of the players facing different directions, orientations and positions on the pitch. This would ensure the custom object detector would be more successful. I created two different datasets from these images. The first dataset was 70 images of the players in blue, and the second dataset was 71 images of the players in white.
 
 ![27](/uploads/5c2fcb2d7256d3ee7d6da06b8ef137d7/27.png)


Figure 27, snippet of the two different datasets for blue and white kits
The next step was to identify where exactly the player is on these images. For this I used a software called Labelimg, which allowed me to load the dataset. Once the dataset was successfully opened, I selected the exact position of the player in the image by clicking and dragging over the player and saving the result. This creates a text file for each image. 

![28](/uploads/ff12ad21ecc474b12a7029a086a3639c/28.png)
  
Figure 28, using Labellmg to select the player

![29](/uploads/20b981db393d508adccdf8f34ef7f90b/29.png)
 
Figure 29, text file created with Labellmg
Once the players had been selected from the images and the text files were created, they were pulled into a .zip file together. The image dataset was now ready to use.
The next step was to train the image dataset online, this was done by using google colab. Google colab allows you to run python scripts and use machine learning libraries. When the dataset is uploaded to google colab it produces a new configurations file. However, once I had acquired this configurations file, I was unsuccessful in integrating it into my work. This would be done by loading the configurations file with the cv2.dnn.readNet function.
Another feature I intended to implement was a birds-eye view of the football pitch, displaying the movement patterns of the players. The players would be represented by blue and white dots depending on the color of their kit. To do this I would need to have a source image and a destination image. The source image would be a screenshot of the footage for the view of the pitch, and the destination image would be a birds-eye view of a football pitch. 

![30](/uploads/d3e55e28dc89d85d52a2686dc74fa721/30.png)
 
Figure 30, destination image birds-eye view of a football pitch. 

Once the relevant images had been acquired, I would use the paint application to open the images and mark the corners of the football pitch in both images. With this mark I would retrieve the pixel value of the location of these corners. Using these pixel values, I would create a separate file to find the matrix of the images by using the cv2.getPerspectiveTransform function. This function would return a matrix which symbolizes the difference between the two pictures. Once the matrix had been found the points would be stored in a float matrix, in variables pts1 and pts2.
 
![31](/uploads/ea11263c21b93a95ce6255b9b667eaad/31.png)

Figure 31, creating matrix with cv2.getPerspectiveTransform
With these two variables I would be able to draw circles on the destination image in my main file, using the cv2.circle function. These circles would represent the players. I was not able to integrate this feature into my work
