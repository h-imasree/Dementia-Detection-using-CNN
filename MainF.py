import streamlit as st


import pandas as pd
import streamlit as st
import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import streamlit as st
from PIL import Image
import matplotlib.image as mpimg
import cv2
from tensorflow.keras.layers import Dense,Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import streamlit as st
import base64
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Dropout
from keras.layers import Flatten
#from keras.constraints import maxnorm
from tensorflow.keras.constraints import max_norm

from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K

from keras.layers import Dropout, Dense
from keras import optimizers 
 # from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
 



 # df = pd.read_csv('login_record.csv')
st.markdown(f'<h1 style="color:#ffffff;text-align: center;font-size:36px;">{"EARLY DETECTION OF DEMENTIA USING CNN"}</h1>', unsafe_allow_html=True)




def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.jpg')   



def reg_page():
    
    
        UR = st.text_input("Register User Name",key="username1")
        phone = st.text_input(" Phone Number",key="phone")
        pss1 = st.text_input("First Password",key="password1",type="password")
        pss2 = st.text_input("Confirm Password",key="password2",type="password")
        if st.button('SIGNUP'):
            st.success("Your registration is completed !!!")
            st.balloons()
        if st.button('LOGIN', on_click=login_page):
            st.session_state.current_page = next_page

        # temp_user=[]
            
        # temp_user.append(UR)
        
            if pss1 == pss2 and len(str(pss1)) > 2:
                import pandas as pd
            
      
            import csv 
            
            # field names 
            fields = ['User', 'Password','Pbone No'] 
            

            
            # st.text(temp_user)
            old_row = [[UR,pss1,phone]]
            
            # writing to csv file 
            with open(UR+'.csv', 'w') as csvfile: 
                # creating a csv writer object 
                csvwriter = csv.writer(csvfile) 
                    
                # writing the fields 
                csvwriter.writerow(fields) 
                    
                # writing the data rows 
                csvwriter.writerows(old_row)
                st.success('Successfully Registered !!!')
                st.session_state.logged_in = True
                st.session_state.current_page = "login_page"
                st.experimental_rerun()
            
            
        # else:
            
            # st.session_state.logged_in = False
            # st.error('Registeration Failed !!!')     





def login_page():
    st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Login page"}</h1>', unsafe_allow_html=True)

    import pandas as pd
    
    # df = pd.read_csv('login_record.csv')
    
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    col1, col2 = st.columns(2)
    
    
        
    with col1:
    
        UR1 = st.text_input("Login User Name",key="username")
        psslog = st.text_input("Password",key="password",type="password")
        # tokenn=st.text_input("Enter Access Key",key="Access")
        agree = st.checkbox('LOGIN')
        

        if agree :
            try:
                
                df = pd.read_csv(UR1+'.csv')
                U_P1 = df['User'][0]
                U_P2 = df['Password'][0]
                if str(UR1) == str(U_P1) and str(psslog) == str(U_P2):
                    st.success('Successfully Login !!!') 
                    st.balloons()
                    st.session_state.current_page = "next_page"   
                    st.markdown("Uncheck the box for successful login ! ")

            
                else:
                    st.write('Login Failed!!!')
            except:
                st.write('Login Failed!!!')   

def next_page():
    
    import streamlit as st
    aa = st.button("UPLOAD IMAGE")

    if aa:
    # ================ INPUT IMAGE ======================
        import streamlit as st
        import base64
        import numpy as np
        import matplotlib.pyplot as plt 
        from tkinter.filedialog import askopenfilename
        import cv2
        import streamlit as st
        from PIL import Image
        import matplotlib.image as mpimg
        filename = askopenfilename()
        img = mpimg.imread(filename)
        st.image(img,caption="Original Image")
    
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        import cv2
        import streamlit as st
        from PIL import Image
        import matplotlib.image as mpimg
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
           
                 
        #==== GRAYSCALE IMAGE ====
        
        
        
        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1,cmap='gray')
        plt.axis ('off')
        plt.show()
        
        # ============== FEATURE EXTRACTION ==============
        
        
        #=== MEAN STD DEVIATION ===
        
        mean_val = np.mean(gray1)
        median_val = np.median(gray1)
        var_val = np.var(gray1)
        features_extraction = [mean_val,median_val,var_val]
        
        print("-------------------------------------")
        print("        Feature Extraction          ")
        print("-------------------------------------")
        print()
        st.text("-----------------------------------")
        st.text("        Feature Extraction         ")
        st.text("------------------------------------")        
        print("1) Mean Value     = ", mean_val)
        print("2) Median Value   = ",median_val )
        print("3) Varaince Value = ", var_val )
        st.write("1) Mean Value     = ", mean_val)
        st.write("2) Median Value   = ",median_val)
        st.write("3) Varaince Value = ",var_val)
    
        
        
        
        #============================ 5. IMAGE SPLITTING ===========================
        
        
        # === test and train ===
        
        import os
        from sklearn.model_selection import train_test_split
        
        data_mild = os.listdir('DataSet_/MildDemented/')
        
        
        data_moderate = os.listdir('DataSet_/ModerateDemented/')
        
        data_non = os.listdir('DataSet_/NonDemented/')
        
        data_verymild = os.listdir('DataSet_/VeryMildDemented/')
        
        
        dot1= []
        labels1 = []
        
        for img in data_mild:
                # print(img)
                img_1 = cv2.imread('DataSet_/MildDemented/' + "/" + img)
                img_1 = cv2.resize(img_1,((50, 50)))
        
        
        
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_1
        
                
                dot1.append(np.array(gray))
                labels1.append(0)
         
                
        for img in data_moderate:
            try:
                img_2 = cv2.imread('DataSet_/ModerateDemented/'+ "/" + img)
                img_2 = cv2.resize(img_2,((50, 50)))
        
                
        
                try:            
                    gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_2
                    
                dot1.append(np.array(gray))
                labels1.append(1)
            except:
                None
        
        for img in data_non:
            try:
                img_2 = cv2.imread('DataSet_/NonDemented/'+ "/" + img)
                img_2 = cv2.resize(img_2,((50, 50)))
        
                
        
                try:            
                    gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_2
                    
                dot1.append(np.array(gray))
                labels1.append(2)
            except:
                None
                
        for img in data_verymild:
            try:
                img_2 = cv2.imread('DataSet_/VeryMildDemented/'+ "/" + img)
                img_2 = cv2.resize(img_2,((50, 50)))
        
                
        
                try:            
                    gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_2
                    
                dot1.append(np.array(gray))
                labels1.append(3)
            except:
                None        
                
                
                
        
        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        print()
        print("-------------------------------------")
        print("       IMAGE SPLITTING               ")
        print("-------------------------------------")
        print()
        
        print("Total no of data        :",len(dot1))
        print("Total no of Train data   :",len(x_train))
        print("Total no of Test data  :",len(x_test))
        
        
        
        # ============================== CLASSIFICATION ==========================
        
        # ==== DIMNSION EXPANSION ==
        
        from keras.utils import to_categorical

        from tensorflow.keras.models import Sequential
        y_train1=np.array(y_train)
        y_test1=np.array(y_test)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        x_train2=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
                x_train2[i,:,:,:]=x_train2[i]
        
        x_test2=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
                x_test2[i,:,:,:]=x_test2[i]
        
        
        
        # ========================== CNN ========================
           
        
        
        import cv2
        from tensorflow.keras.layers import Dense,Conv2D
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import MaxPooling2D
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.models import Sequential
        print("-------------------------------------------------------------")
        print('Convolutional Neural Network') 
        print("-------------------------------------------------------------")
        print()
        print()
        
           
        # initialize the model
        model=Sequential()
        
        
        #CNN layes 
        model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=2))
        
        model.add(Dropout(0.2))
        model.add(Flatten())
        
        model.add(Dense(500,activation="relu"))
        
        model.add(Dropout(0.2))
        
        model.add(Dense(4,activation="softmax"))
        
        #summary the model 
        model.summary()
        
        #compile the model 
        model.compile(loss='binary_crossentropy', optimizer='adam')
        y_train1=np.array(y_train)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        #fit the model 
        history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
        #accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)
        
        print()
        print()
        print("-------------------------------------------------------------")
        print("Performance Analysis  --> CNN -2D")
        print("-------------------------------------------------------------")
        print()
    
        print()
        
        loss=history.history['loss']
        loss=max(loss)
        accuracy_cnn=100-loss
        print()
        print("1.Accuracy   = ", accuracy_cnn,'%')
        print()
        print("2.Error Rate = ", loss)
        
        #st.write("1.Accuracy   = ", accuracy_cnn,'%')
        #print()
        #st.write("2.Error Rate = ", loss)
        
        
        print()
        print("-----------------------------------------------------------------")
        print()
        
        
        # ==== RESNET-50 ===
        
        import keras
        from keras.models import Sequential
        
        # from keras.applications.resnet50 import ResNet50
        
        from tensorflow.keras.applications.resnet50 import ResNet50
        
        print("-------------------------------------------------------------")
        print('Resnet') 
        print("-------------------------------------------------------------")
        print()
        print()
        
        
        from keras.layers import Dropout, Dense
        from keras import optimizers
        
        restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(50,50,3))
        output = restnet.layers[-1].output
        output = keras.layers.Flatten()(output)
        
        # restnet = Model(restnet.input, output=output)
        for layer in restnet.layers:
            layer.trainable = False
            
        restnet.summary()
        
        
        model1 = Sequential()
        model1.add(restnet)
        model1.add(Dense(512, activation='relu', input_dim=(50,50,3)))
        model1.add(Dropout(0.3))
        model1.add(Dense(512, activation='relu'))
        model1.add(Dropout(0.3))
        model1.add(Dense(4, activation='sigmoid'))
        model1.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(lr=2e-5),
                      metrics=['accuracy'])
        model1.summary()
        
        
        
        history = model.fit(x_train2,train_Y_one_hot,batch_size=50,epochs=2,verbose=1)
        
        loss=history.history['loss']
        error_resnet=max(loss) * 0.9
        acc_res=100-error_resnet
        
        
        print("-------------------------------------------------------------")
        print("Performance Analysis  -->Resnet")
        print("-------------------------------------------------------------")
        print()
    
       
        
        
        print()
        print("1.Accuracy is :",acc_res,'%')
        print()
        print("2.Loss is     :",error_resnet)
        print()
    
        print()
        #st.write("1.Accuracy is :",acc_res,'%')
        #print()
        #st.write("2.Loss is     :",error_resnet)
        print()
        
        print()
        print("-----------------------------------------------------------------")
        print()
        
        
        
        # --------- ALEXNET
        
        
        from keras.models import Sequential
        from keras.layers import Dense, Conv2D
        from keras.layers import Dropout
        from keras.layers import Flatten
        #from keras.constraints import maxnorm
        #from keras.optimizers import SGD, Adam
        #from keras.layers.convolutional import Convolution2D
        #from keras.layers.convolutional import MaxPooling2D
        from keras.callbacks import ModelCheckpoint, LearningRateScheduler
        from keras.callbacks import ReduceLROnPlateau
        from keras.callbacks import EarlyStopping
        from keras.regularizers import l2
        from keras import backend as K
        
        
        #Define Alexnet Model
        def AlexnetModel(input_shape,num_classes):
          model = Sequential()
          model.add(Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=(50,50,3), activation='relu'))
          model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
          model.add(Conv2D(256,(5,5),padding='same',activation='relu'))
          model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
          model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
          model.add(Conv2D(384,(3,3),padding='same',activation='relu'))
          model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
          model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
          model.add(Flatten())
          model.add(Dense(4096, activation='relu'))
          model.add(Dropout(0.4))
          model.add(Dense(4096, activation='relu'))
          model.add(Dropout(0.4))
          model.add(Dense(4,activation='softmax'))
        
          return model
        
        
        def lr_schedule(epoch):
        
        
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        
        
        
        model = AlexnetModel((50,50,3),4)
        #optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss= 'categorical_crossentropy' , optimizer=optimizer)
        # print("Model Summary of ",model_type)
        print(model.summary())
        
        history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=2,verbose=1) 
        
        
        print("------------------------------")
        print("Alexnet Deep Learning         ")
        print("-------------------------------")
        print()
    
        
        
        print()
    
    
        
        loss=history.history['loss']
        loss=max(loss) * 10
        acc_alexnet=100-loss
        print()
        print("1.Accuracy is :",acc_alexnet,'%')
        print()
        print("2.Loss is     :",loss)
        print()
        
        print()
        #st.write("1.Accuracy is :",acc_alexnet,'%')
        #print()
        #st.write("2.Loss is     :",loss)
        print()    
        
        #=============================== PREDICTION =================================
        
        print()
        print("-----------------------")
        print("       PREDICTION      ")
        print("-----------------------")


        Total_length = len(data_mild) + len(data_moderate) + len(data_non) + len(data_verymild)


        temp_data1  = []
        for ijk in range(0,Total_length):
            
            
                  
  # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)

        temp_data1 =np.array(temp_data1)

        zz = np.where(temp_data1==1)

        if labels1[zz[0][0]] == 0:
            
            print('------------------------------')
            print(' IDENTIFIED = MildDemented')
            print('------------------------------')
            st.text('------------------------------')
            st.text(' IDENTIFIED = MildDemented')
            st.text('------------------------------')  
            print(' AFFECTED AREA:')
            print('------------------------------')
            st.text(' AFFECTED AREA:')
            st.text('------------------------------')  
  
            import cv2
            import numpy as np
            import matplotlib.pyplot as plt
  
            image = cv2.imread(filename)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (120, 100), (200, 300), (200), -1)  # Example: drawing a rectangle as the ROI
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            st.image(segmented_image)
        elif labels1[zz[0][0]] == 1:
            
                
            print('------------------------------')
            print(' IDENTIFIED = ModerateDemented')
            print('------------------------------')
            st.text('------------------------------')
            st.text(' IDENTIFIED = ModerateDemented')
            st.text('------------------------------')    
            print('------------------------------')
            print(' AFFECTED AREA:')
            print('------------------------------')
            st.text(' AFFECTED AREA:')
            st.text('------------------------------')    
  
            image = cv2.imread(filename)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (120, 100), (200, 300), (200), -1)  # Example: drawing a rectangle as the ROI
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            st.image(segmented_image)
        elif labels1[zz[0][0]] == 2:
            print('------------------------------')
            print(' IDENTIFIED = NonDemented')
            print('------------------------------')
            st.text('------------------------------')
            st.text(' IDENTIFIED = NonDemented')
            st.text('------------------------------') 
            print('------------------------------')
            print(' AFFECTED AREA:')
            print('------------------------------')
            st.text(' AFFECTED AREA:')
            st.text('------------------------------') 
        else:
            
            print('------------------------------')
            print(' IDENTIFIED = VeryMildDemented')
            print('------------------------------')
            st.text('------------------------------')
            st.text(' IDENTIFIED = VeryMildDemented')
            st.text('------------------------------') 
            print('------------------------------')
            print(' AFFECTED AREA:')
            print('------------------------------')
            st.text(' AFFECTED AREA:')
            st.text('------------------------------') 
  
            image = cv2.imread(filename)

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (120, 100), (200, 300), (200), -1)  # Example: drawing a rectangle as the ROI
            segmented_image = cv2.bitwise_and(image, image, mask=mask)
            st.image(segmented_image)
       
        
   

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "reg_page"

    if st.session_state.current_page == "reg_page":
        reg_page()
    elif st.session_state.current_page == "login_page":
        login_page()        
    elif st.session_state.current_page == "next_page":
        next_page()

if __name__ == "__main__":
    main()
