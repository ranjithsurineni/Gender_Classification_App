import os
import cv2
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request 
import matplotlib.image as matimg


UPLOAD_FOLDER = 'static/uploads'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def contact():
    return render_template('contact.html')

def genderapp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        #save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) #save image into upload folder
        #get predictaions
        pred_images, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg' #save image in static folder
        cv2.imwrite(f'./static/predicts/{pred_filename}',pred_images) #save image in static folder
       
        
        #generate report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi'] #grayscale image (array)
            eig_image = obj['eig_img'].reshape(100,100) #eigen image
            gender_name = obj['prediction_name'] #prediction name
            score = round(obj['score']*100,2) # Probability score

            #save grayscale image and eigen image in predicts folder
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predicts/{gray_image_name}',gray_image, cmap='gray')
            matimg.imsave(f'./static/predicts/{eig_image_name}',eig_image , cmap='gray')
            
            #save report
            report.append([gray_image_name,
                           eig_image_name,
                           gender_name,
                           score])
                           
        return render_template('gender.html', fileupload=True,report=report)  #Post Request

    return render_template('gender.html', fileupload=False) #Get request