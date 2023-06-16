import cv2

# Load the cow Haar cascade XML file
cow_cascade = cv2.CascadeClassifier('cow_cascade.xml')

# Open the video file
video = cv2.VideoCapture('videoplayback.mp4')

while True:
    # Read a frame from the video
    ret, frame = video.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cows in the frame
    cows = cow_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles and add text for each cow detected
    for (x, y, w, h) in cows:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Cow', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Cow Detection', frame)

    # Check for the 'q' key to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()
