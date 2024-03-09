# Dự đoán giới tính theo họ tên(VN Edition)

## Giới thiệu
Ứng dụng này được tạo ra để dự đoán giới tính dựa trên họ và tên của một người sử dụng mô hình Naive Bayes và CountVectorizer đã được huấn luyện trước. Người dùng có thể nhập họ tên của họ vào ô văn bản và nhấn vào nút "Dự đoán" để nhận kết quả.

## Cài đặt và chạy ứng dụng
1. Đảm bảo bạn đã cài đặt Python trên máy tính của mình.
2. Cài đặt các thư viện cần thiết bằng cách chạy `pip install -r requirements.txt`.
3. Mở terminal hoặc command prompt và điều hướng đến thư mục chứa mã nguồn của ứng dụng.
4. Chạy ứng dụng bằng cách gõ `streamlit run app.py`.

## Cách sử dụng
1. Khi ứng dụng được khởi chạy, trình duyệt sẽ mở và hiển thị giao diện người dùng.
2. Nhập họ tên của bạn vào ô văn bản được cung cấp.
3. Nhấn nút "Dự đoán" để xem kết quả.
4. Kết quả sẽ hiển thị dưới dạng "Giới tính dự đoán: Nam" hoặc "Giới tính dự đoán: Nữ" tùy thuộc vào dự đoán của mô hình.

## Lưu ý
- Ứng dụng này chỉ cung cấp dự đoán giới tính dựa trên họ và tên dành cho tiếng việt, không có chứng chỉ về độ chính xác tuyệt đối.
- Kết quả có thể không chính xác 100% và chỉ nên được coi là một dự đoán mang tính chất tham khảo.

Đây là một ứng dụng đơn giản và chỉ mang tính chất minh họa về cách sử dụng mô hình học máy để dự đoán giới tính.
