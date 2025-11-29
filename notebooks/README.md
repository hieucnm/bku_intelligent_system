
# 📘 Loan Default Prediction — ML Project

Dự án này xây dựng một hệ thống Machine Learning hoàn chỉnh cho bài toán **Loan Default Prediction** (dự đoán khả năng vỡ nợ), bao gồm:

* Khám phá dữ liệu
* Huấn luyện mô hình (LightGBM)
* Triển khai giao diện đa - stakeholder bằng  Gradio

Hệ thống hỗ trợ 4 nhóm người dùng: **Loan Officer**, **Risk Manager**, **Data Scientist**, và **End User**.

---

# 📂 Project Structure

```
.
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_Train.ipynb
│   └── 3_Deploy.ipynb
├── models/v2025-11-27
├── data/
│   ├── Loan_Default.csv/           # Dữ liệu gốc
│   └── test_data.csv  # Dataset mẫu dùng để demo Upload
├── requirements.txt
└── README.md
```

---

# ▶️ Notebook Execution Order

Để chạy project theo đúng pipeline, hãy thực hiện theo thứ tự:

### **1. (Tuỳ chọn) `1_EDA.ipynb`**

* Khám phá dữ liệu
* Kiểm tra missing values, outliers
* Phân tích phân phối và thống kê mô tả

### **2. `2_Train.ipynb`**
* Train 5 mô hình LightGBM
* Lưu model vào thư mục `./models/`

### **3. `3_Deploy.ipynb`**

* Load mô hình đã train
* Tạo UI Gradio
* Demo

---

# 🔑 Stakeholder Login Accounts

Ứng dụng có 4 tab, mỗi tab tương ứng với một loại stakeholder.
Khi chạy UI, bạn có thể đăng nhập bằng các tài khoản mẫu sau:

| Stakeholder  | Username  | Password |
| ------------ | --------- | -------- |
| Loan Officer | `officer` | `123`    |
| Risk Manager | `risk`    | `123`    |
| BA / DA / DS | `ds`      | `123`    |
| End User     | `user`    | `123`    |

> ⚠️ *Lưu ý:* Đây hoàn toàn là public data phục vụ demo, không có cơ chế bảo mật thực tế.

---

# 🧪 Demo Dataset (Upload)

Dự án cung cấp 1 file dataset mẫu để demo các tính năng **Upload dataset** trong UI:

```
./data/test_data.csv
```

File này có thể được dùng ở các tab:

* Loan Officer (tìm kiếm hồ sơ)
* Risk Manager (lọc theo date và phân tích rủi ro)
* Data Scientist (đánh giá mô hình)

Ở tab End User, để demo việc nhập số điện thoại để xem kết quả duyệt hồ sơ, trong khi dataset này không có số điện thoại, nhóm đã tạo 2 số điện thoại mẫu hoàn toàn không có thực là `0909123456` và `0909654321`. Người dùng chỉ có thể dùng 2 số này để demo. Profile của 2 số này được lấy ngẫu nhiên trong data gốc và lưu tại `./data/test_processed_user_sample.csv`

---

# 🚀 Features

### ✔ Loan Officer Dashboard

* Tra cứu hồ sơ theo số điện thoại / ID
* Hiển thị thông tin khách hàng
* Dự đoán xác suất vỡ nợ + giải thích (SHAP / Feature importance)

### ✔ Risk Manager Dashboard

* Upload dataset và lọc theo ngày
* Histogram score, phân phối nhãn dự đoán
* Weekly default rate line chart
* Tóm tắt rủi ro theo thời gian

### ✔ Data Scientist Dashboard

* Upload dataset đánh giá mô hình
* ROC Curve, Confusion Matrix
* Accuracy, AUC, Recall, Precision
* Feature importance

---

# 🖥️ Run the App

### Setup:
Cài đặt Python version 3.13

Cài đặt  thư viện
```bash
pip install -r requirements.txt
```

### Chạy UI:

1. Chạy notebook `notebooks/3_Deploy.ipynb` từ đầu đến cuối, và mở đường dẫn localhost được xuất ra tại ô cuối cùng của notebook (thông thường sẽ là `http://127.0.0.1:7860`), nó sẽ mở Gradio UI của chương trình lên.
2. Đăng nhập stackholder theo account và password ở trên
3. Upload datasets hoặc nhập số điện thoại và trải nghiệm.

---

# 💬 Contact
- cnmhieu.sdh242@hcmut.edu.vn
- tdhung.sdh242@hcmut.edu.vn