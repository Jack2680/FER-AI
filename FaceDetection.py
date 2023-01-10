import face_recognition
from PIL import Image

imagePath = 'D:/GroupImage.png'

image = face_recognition.load_image_file(imagePath)
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:

    top, right, bottom, left = face_location

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()



