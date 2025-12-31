import cv2
import os

#start video capture
cap = cv2.VideoCapture(0)

#directory to staring the images
directory = 'Image/'

while True:
    _, frame = cap.read()
    #dictionary to count nymber of images in each subdir
    count={
        'a': len(os.listdir(directory +'A')),
        'b': len(os.listdir(directory +'B')),
        'c': len(os.listdir(directory +'C')),
    }
    #get dimentions of each captured image
    row = frame.shape[1]
    col = frame.shape[0]

    #draw a white rectangel to show capture area
    cv2.rectangle(frame, (0,40), (300, 400), (255,255,255), 2)

    #display the capture region separately 
    cv2.imshow('data',frame)
    cv2.imshow('ROI', frame[40:400, 0:300])

    #crop
    frame = frame[40:400, 0:300]

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.png', frame)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.png', frame)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.png', frame)


#release the video ccapture device
cap.release()
cv2.destroyAllWindows()
