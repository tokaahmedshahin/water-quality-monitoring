#import libraries
import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle
import pandas as pd
#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file) hena ana bft7 el model file and save it in variable 

model = pickle.load(open(r'model/model.pkl', 'rb'))
# Define a list of class names corresponding to the model's output classes
class_names = ['carpio','catla', 'koi','magur', 'pangasius', 'prawn','rui','shing','shrimp','silverCarp','tilapia']  


#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data  (ha5od data eli b3tha el web)

#POST: Used to send HTML form data to the server. (result of classification)
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output

'''
delw2ty me7taga a7ded eli eli hy7sl lama tegili data
 h7ded h3ml eh w ha5od ezay el data mn form eli path bta3o esmo /predict 
    /predict -------->endpoint eli hat handle HTTP POST requests
'''
@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()

        # Create a Pandas DataFrame from JSON data
        df = pd.DataFrame(json_data)
        # Extract features from the DataFrame
        int_features = df.values.flatten().astype(float)  # Flatten and convert to float
        #exclude tank number
        modified_int_features=[int_features[1],int_features[2],int_features[3]]
        modified_int_features = [np.array(int_features)]  # Convert to the form [[a, b, c, ...]] for input to the model
        prediction = model.predict_proba(modified_int_features)  # features Must be in the form [[a, b]]
        # Filter probabilities to keep only those greater than 50%
        threshold = 0.50
        filtered_proba = [p for p in prediction[0] if p > threshold]
        
        # If no probabilities are greater than 50%, handle this case
        if not filtered_proba:
            predicted_class_index = None
            predicted_class_name = "the water sample is not suitable for any fish"
        else:
            # Get the index of the maximum probability among the filtered ones
            max_proba = max(filtered_proba)
            predicted_class_index = np.where(prediction[0] == max_proba)[0][0]
            predicted_class_name = class_names[predicted_class_index]

        
        # Create a JSON response
        response = {
                 'tank_number':int_features[0],
                'predicted_class': predicted_class_name,
                
            }
        return jsonify(response)
    except Exception as e:
        print(str(e))
        return jsonify({'error':str(e)})
    


    
#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()
