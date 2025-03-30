import streamlit as st
import os
import pickle
import re
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Dự đoán giới tính từ tên tiếng Việt",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
        background-color: #4267B2;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        font-size: 1rem;
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .male-box {
        background-color: #D4E6F1;
        border-left: 5px solid #3498DB;
    }
    .female-box {
        background-color: #F8D7DA;
        border-left: 5px solid #E83E8C;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #7F8C8D;
    }
</style>
""", unsafe_allow_html=True)

# Function to preprocess name
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
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "save_models/svm_count.pkl")
    vectorizer_path = os.path.join(current_dir, "save_models/vectorizer_count.pkl")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

model, vectorizer, models_loaded = load_models()

def predict_gender(name):
    if not models_loaded:
        return None, None
        
    # Preprocess the name
    processed_name = preprocess_name(name)
    
    # Extract middle and first name components (which the model uses)
    family_name, middle_name, first_name = extract_name_components(processed_name)
    middle_first = (middle_name + ' ' + first_name).strip()
    
    # Vectorize the name using the loaded CountVectorizer
    name_vectorized = vectorizer.transform([middle_first])
    
    # Predict gender and get probability
    prediction = model.predict(name_vectorized)[0]
    probabilities = model.predict_proba(name_vectorized)[0]
    
    # Get confidence score
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    return prediction, confidence, family_name, middle_name, first_name

# Function to create a gauge chart for confidence
def create_confidence_gauge(confidence, gender):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=True))
    
    # Determine colors based on gender
    if gender == 0:  # Female
        color = "#E83E8C"
    else:  # Male
        color = "#3498DB"
    
    # Background ring
    ax.bar(x=0, height=1, width=np.pi*2, bottom=0, color="#E0E0E0", alpha=0.5)
    # Value ring
    ax.bar(x=0, height=1, width=np.pi*2 * confidence, bottom=0, color=color)
    
    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    
    # Add text in the middle
    ax.text(0, 0, f"{confidence:.1%}", ha='center', va='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Generate example names
def generate_example_names():
    female_examples = [
        "Nguyễn Thị An", 
        "Trần Kim Liên", 
        "Vũ Hoàng Yến",
        "Phạm Thu Thảo",
        "Lê Quỳnh Anh"
    ]
    
    male_examples = [
        "Nguyễn Văn Minh",
        "Trần Quốc Bảo",
        "Phạm Đức Hùng",
        "Vũ Tuấn Anh",
        "Hoàng Mạnh Huy"
    ]
    
    return female_examples, male_examples

# Sidebar
with st.sidebar:
    st.image("gender.jpg" if os.path.exists("gender.jpg") else None, use_column_width=True)
    st.header("Thông tin")
    st.markdown("""
    Ứng dụng này sử dụng mô hình học máy để dự đoán giới tính dựa trên tên tiếng Việt.
    
    **Cách sử dụng:**
    1. Nhập họ tên đầy đủ vào ô
    2. Nhấn nút "Dự đoán giới tính"
    3. Xem kết quả và phân tích chi tiết
    
    **Lưu ý:** Độ chính xác phụ thuộc vào dữ liệu huấn luyện. Một số tên có thể dùng cho cả nam và nữ.
    """)
    
    st.header("Cách hoạt động")
    st.markdown("""
    Mô hình sử dụng kỹ thuật học máy SVM (Support Vector Machine) để phân loại dựa trên các mẫu tên trong tập dữ liệu huấn luyện.
    
    Ứng dụng chủ yếu phân tích phần tên đệm và tên, vì đây là phần thường thể hiện giới tính rõ nhất trong tên tiếng Việt.
    """)

# Main app
def main():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🧠 Dự đoán giới tính từ tên tiếng Việt")
        st.markdown("Nhập họ và tên đầy đủ để dự đoán giới tính dựa trên mô hình học máy.")
    
    # Name input
    name_input = st.text_input("Họ và tên đầy đủ:", placeholder="Ví dụ: Nguyễn Văn An", key="name_input")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Dự đoán giới tính")
    
    # Example names
    female_examples, male_examples = generate_example_names()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Ví dụ tên nữ:")
        for example in female_examples:
            if st.button(example, key=f"female_{example}"):
                st.session_state.name_input = example
                predict_button = True
    
    with col2:
        st.markdown("##### Ví dụ tên nam:")
        for example in male_examples:
            if st.button(example, key=f"male_{example}"):
                st.session_state.name_input = example
                predict_button = True
    
    # Error handling
    if not models_loaded:
        st.error("Không thể tải mô hình. Vui lòng kiểm tra lại thư mục 'save_models'.")
        return
    
    # Prediction logic
    if predict_button or ('name_input' in st.session_state and st.session_state.name_input):
        user_input = st.session_state.name_input if 'name_input' in st.session_state else name_input
        
        if user_input:
            with st.spinner("Đang phân tích..."):
                # Get prediction
                gender, confidence, family, middle, first = predict_gender(user_input)
                
                # Display result
                st.markdown("### Kết quả phân tích")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    if gender == 0:
                        st.markdown(f"""
                        <div class="result-box female-box">
                            <h2>👩 Giới tính: Nữ</h2>
                            <p>Phân tích cho tên <b>{user_input}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box male-box">
                            <h2>👨 Giới tính: Nam</h2>
                            <p>Phân tích cho tên <b>{user_input}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### Độ tin cậy")
                    fig = create_confidence_gauge(confidence, gender)
                    st.pyplot(fig)
                
                # Name components
                st.markdown("### Chi tiết phân tích")
                
                components_df = pd.DataFrame({
                    'Thành phần': ['Họ', 'Tên đệm', 'Tên', 'Phần dùng để dự đoán'],
                    'Giá trị': [family, middle, first, f"{middle} {first}".strip()]
                })
                
                st.table(components_df)
                
                # Gender characteristics
                st.markdown("### Đặc điểm tên theo giới tính")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    #### Tên nữ thường có:
                    - Tên đệm: Thị, Kim, Ngọc, Thu, Thanh, Thùy
                    - Tên: Anh, Linh, Hương, Thảo, Lan, Hà, Yến, Ngọc, Trang
                    """)
                
                with col2:
                    st.markdown("""
                    #### Tên nam thường có:
                    - Tên đệm: Văn, Hữu, Đức, Quốc, Mạnh, Tuấn
                    - Tên: Anh, Minh, Hùng, Đức, Hải, Bảo, Hoàng, Phong, Dũng
                    """)
                
                # Explanation
                st.markdown("### Giải thích kết quả")
                
                if confidence > 0.9:
                    st.success(f"Kết quả có độ tin cậy rất cao ({confidence:.1%}). Mô hình rất tự tin với dự đoán này.")
                elif confidence > 0.7:
                    st.info(f"Kết quả có độ tin cậy khá cao ({confidence:.1%}). Mô hình khá tự tin với dự đoán này.")
                else:
                    st.warning(f"Kết quả có độ tin cậy trung bình ({confidence:.1%}). Tên này có thể dùng cho cả nam và nữ.")
        else:
            st.warning("Vui lòng nhập họ tên để dự đoán.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 - Ứng dụng Dự đoán giới tính từ tên tiếng Việt</p>
    </div>
    """, unsafe_allow_html=True)

# Add import for confidence gauge
import numpy as np

# Run the app
if __name__ == "__main__":
    main()
