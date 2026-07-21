import pandas as pd
import joblib

def predict_price(rating, loves_count, is_fragrance):
    loaded_model = joblib.load("sephora_model.pkl")
    input_data = pd.DataFrame([[rating, loves_count, is_fragrance]], columns=["rating", "loves_count", "primary_category_Fragrance"])
    prediction = loaded_model.predict(input_data)
    return prediction[0]
print(predict_price(4.5, 2000, True))    
print(predict_price(3.0, 500, False))   
print(predict_price(5.0, 50000, True))  