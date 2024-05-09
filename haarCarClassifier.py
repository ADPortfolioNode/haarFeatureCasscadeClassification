import urllib.request
import cv2
from matplotlib import pyplot as plt

# Create a function that cleans up and displays the image
def plt_show(image, title="Plotting image", gray=False):
    """
    Display the image.
    
    Parameters:
    image (numpy.ndarray): The image to display.
    title (str): The title of the image (default: "").
    gray (bool): Whether to convert the image to grayscale (default: False).
    """
    temp = image 
    
    # Convert to grayscale images
    if gray is False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    
    # Remove axes ticks
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp, cmap='gray')
    plt.show()

def detect_obj(image):
    """
    Detect objects in the image and draw rectangles around them.
    
    Parameters:
    image (numpy.ndarray): The image to detect objects in.
    """
    # Clean up the image
    plt_show(image, "detecting ....")
    
    # Detect the car in the image
    object_list = detector.detectMultiScale(image)
    print(object_list)
    
    # For each car, draw a rectangle around it
    for obj in object_list: 
        (x, y, w, h) = obj
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 50, 75), 9) # Line thickness
    
    # View the image
    plt_show(image,"lines drawn")

# Load the Haar cascade XML file
haarcascade_url = 'https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/cars.xml'
HAAR_NAME = "cars.xml"
urllib.request.urlretrieve(haarcascade_url, HAAR_NAME)

# Load the detector
detector = cv2.CascadeClassifier(HAAR_NAME)

# Read the image
image_url = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/CV0101/Dataset/car-road-behind.jpg"
IMAGE_NAME = "car-road-behind.jpg"
urllib.request.urlretrieve(image_url, IMAGE_NAME)
image = cv2.imread(IMAGE_NAME)

plt_show(image,IMAGE_NAME)

# Detect objects in the image
detect_obj(image) 
# /////////////////////////////////////////////# 
plt_show(image, title="Detected Objects")
 


# Test with your own image
my_image = cv2.imread("s-original.jpg")
plt_show(my_image,"uploaded by you")
detect_obj(my_image)
 

plt_show(my_image,"final")

print(">>>>>>>>>>>>>>>>>End of Line<<<<<<<<<<<<<<<")