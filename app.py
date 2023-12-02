import streamlit as st
import os
import joblib

# Load the Naive Bayes model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "NB_model.pkl")
cv_path = os.path.join(current_dir, "count.pkl")

# Load the CountVectorizer
model = joblib.load(model_path)
cv = joblib.load(cv_path)

def predict_gender(name):
    # Preprocess the input name
    name = [name]

    # Vectorize the name using the loaded CountVectorizer
    name_vectorized = cv.transform(name)

    # Make the gender prediction
    prediction = model.predict(name_vectorized)

    return prediction[0]

# Streamlit app
def main():
    st.title("Dự đoán giới tính theo họ tên")

    # Display an image
    image_path = os.path.join(current_dir, "gender.jpg")
    st.image(image_path, use_column_width=True)
    # Input box for the user to enter their name
    user_input = st.text_input("Nhập họ tên của bạn:")

    if st.button("Dự đoán"):
        if user_input:
            # Get the predicted gender
            gender = predict_gender(user_input)

            # Display the result
            if gender == 0:
                st.success("Giới tính dự đoán: Nữ")
            else:
                st.success("Giới tính dự đoán: Nam")
        else:
            st.warning("Vui lòng nhập họ tên.")

if __name__ == "__main__":
    main()
