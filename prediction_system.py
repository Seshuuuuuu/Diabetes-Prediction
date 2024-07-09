import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/ADMIN/Downloads/trained_model.sav','rb'))

input_data = (1,85,66,29,0,26.6,0.351,31)
input_data_as_array = np.asarray(input_data)
reshaped_input_data = input_data_as_array.reshape(1,-1)
result = loaded_model.predict(reshaped_input_data)

if(result == 1):
  print('The patient is Diabetic')
else:
  print('The patient is Not Diabetic')