import os
import base64
import pickle
import streamlit as st

from streamlit_option_menu import option_menu
from keras.models import load_model
from PIL import ImageOps, Image
from PIL import Image
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
     
    
def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    #index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score


   
# Set page configuration
st.set_page_config(page_title="MediMind AI",
                   layout="wide",
                   page_icon="🩺")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))

heart_disease_model = pickle.load(open("heart_disease_model.sav", 'rb'))

parkinsons_model = pickle.load(open("parkinsons_model.sav", 'rb'))

# sidebar for navigation
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('MediMind AI',
                          
                          [ 'Home',
                           'Brain Tumor Classification',
                           'Pneumonia Classification',
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction','About Us'],
                          icons=['activity','heart','person'],
                          default_index=0)
    


#home pages
if (selected == 'Home'):
    set_background('Website.png')
    # page title
    #st.title('Home')
    
    st.write("# Welcome to MediMind AI WEBSITE 👋")

    #st.sidebar.success("Select a demo above.")
    
    st.markdown(
        """
        
        MediMind AI is a website where users can diagnose diseases. Basically, this website will be a helpful platform where users will easily diagnose diseases by providing the required information and get an instant result.
        
        ### Why we do this?
        Bangladesh faces significant healthcare challenges, including a high prevalence of various diseases and limited access to medical facilities, especially in rural areas. Early detection and diagnosis of diseases play a crucial role in improving healthcare outcomes. Machine learning (ML) and artificial intelligence (AI) have shown promise in automating disease detection, reducing the burden on healthcare professionals, and increasing the chances of early intervention. Machine learning and artificial intelligence (AI) can make a huge chance in this field but in our country, most people don’t know about it clearly. We try to make it easy and simple. We try our best to make our website in the Bangla language also.
        ### Our Vision:
        -Apply AI for immediate disease diagnosis.
        
        -Explore advanced machine learning techniques and models to enhance diagnostic capabilities.
        
        -Invest in ongoing research and development efforts to improve disease prediction accuracy.
        
        -Incrase health awareness.
        

    """
    )
    
    
    #image
    #st.image('D:ML\pages\MDAI.jpg',caption='hallo')
    
    
#Brain Tumor Classification

if (selected == 'Brain Tumor Classification'):
    set_background('Website.png')
    # set title
    st.title('Brain Tumor Classification')

    # set header
    st.header('Please upload a MRI image of brain')

    # upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # load classifier
    model = load_model('brainTumer_classifier.h5')

    # load class names
    with open('brainTumer_labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    

        
    

#Pneumonia Detection

if (selected == 'Pneumonia Classification'):
    set_background('Website.png')
    # set title
    st.title('Pneumonia classification')

    # set header
    st.header('Please upload a chest X-ray image')

    # upload file
    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # load classifier
    model = load_model('pneumonia_classifier.h5')

    # load class names
    with open('pneumonia_labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()

    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
    



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    set_background('Website.png')
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        set_background('Website.png')
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    set_background('Website.png')
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        set_background('Website.png')
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    set_background('Website.png')
    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        set_background('Website.png')
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
if (selected == 'About Us'):
    set_background('Website.png')
    # page title
    #st.title('Home')
    
    st.write("# About Us 👋")

    #st.sidebar.success("Select a demo above.")
    
    st.markdown(
        """
        
        ### MediMind AI website creared Hanzala & Rayhan.
        
        ### Rayhan Mahmud Ansari
        
        Student of CSE in Sylhet Engineering College.
            
        Phone:01700000000
            
        Email: rayhan@gmai.com  
        
        ### Hanzala Sayed Abdullah
        
        Student of CSE in Sylhet Engineering College.
        
        Phone:01700000000
        
        Email: Hanzala@gmail.com
        

    """
    )
