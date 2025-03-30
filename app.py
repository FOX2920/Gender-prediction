import streamlit as st
import os
import pickle
import re

# Function to preprocess name (from the original model code)
def preprocess_name(name):
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# Function to extract name components
def extract_name_components(name):
    parts = name.split()
    if len(parts) >= 3:
        family_name = parts[0]
        first_name = parts[-1]
        middle_name = ' '.join(parts[1:-1])
    elif len(parts) == 2:
        family_name = parts[0]
        first_name = parts[1]
        middle_name = ""
    else:
        family_name = ""
        middle_name = ""
        first_name = name
    return family_name, middle_name, first_name

# Load the SVM model and CountVectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "saved_models/svm_count.pkl")
vectorizer_path = os.path.join(current_dir, "saved_models/vectorizer_count.pkl")

# Load the model and vectorizer using pickle
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

def predict_gender(name):
    # Preprocess the name
    processed_name = preprocess_name(name)
    
    # Extract middle and first name components (which the model uses)
    _, middle_name, first_name = extract_name_components(processed_name)
    middle_first = (middle_name + ' ' + first_name).strip()
    
    # Vectorize the name using the loaded CountVectorizer
    name_vectorized = vectorizer.transform([middle_first])
    
    # Predict gender and get probability
    prediction = model.predict(name_vectorized)[0]
    probabilities = model.predict_proba(name_vectorized)[0]
    
    # Get confidence score
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    return prediction, confidence

# Streamlit app
def main():
    st.title("Dự đoán giới tính theo họ tên tiếng Việt")
    
    # Display an image if it exists
    image_path = os.path.join(current_dir, "gender.jpg")
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    
    # Input box for the user to enter their name
    user_input = st.text_input("Nhập họ tên của bạn:")
    
    if st.button("Dự đoán"):
        if user_input:
            # Get the predicted gender and confidence
            gender, confidence = predict_gender(user_input)
            
            # Display the result with confidence
            if gender == 0:
                st.success(f"Giới tính dự đoán: Nữ (Độ tin cậy: {confidence:.2%})")
            else:
                st.success(f"Giới tính dự đoán: Nam (Độ tin cậy: {confidence:.2%})")
                
            # Show the name components that were used for prediction
            with st.expander("Chi tiết phân tích"):
                processed_name = preprocess_name(user_input)
                family, middle, first = extract_name_components(processed_name)
                st.write(f"Họ: {family}")
                st.write(f"Tên đệm: {middle}")
                st.write(f"Tên: {first}")
                st.write(f"Phần được sử dụng để dự đoán: {middle + ' ' + first}")
        else:
            st.warning("Vui lòng nhập họ tên.")

if __name__ == "__main__":
    main()
