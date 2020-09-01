import cv2
import mtcnn
import numpy as np
from PIL import Image
import cv2

def save_img():
    key=cv2.waitKey()
    webcam=cv2.VideoCapture(0)
    while True:
        try:
            check,frame=webcam.read()
            cv2.imshow("capturing",frame)
            key=cv2.waitKey(1)
            if key==ord('s'):
                cv2.imshow("captured image",frame)
                pixels=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                detector=mtcnn.MTCNN()
                results=detector.detect_faces(pixels)
                x1,y1,width,height=results[0]['box']
                x1,y1=abs(x1),abs(y1)
                x2,y2=x1+width,y1+height
                #face=pixels[y1:y2,x1:x2]
                face=pixels[y1-15:y2+15,x1-15:x2+15]
                image=Image.fromarray(face)
                image=image.resize((96,96))
                face=np.array(image)
                opencvImage=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                n=input("Enter name:")
                cv2.imwrite(filename="./images/"+n+".jpg",img=opencvImage)
                print("image saved!")
                webcam.release()
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("off...")
                webcam.release()
                cv2.destroyAllWindows()
                break
                
        except(KeyboardInterrupt):
            print("off...")
            webcam.release()
            cv2.destroyAllWindows()
            break
if __name__=="__main__":
    save_img()
