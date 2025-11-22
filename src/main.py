import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, io
from datetime import datetime


# -------------------------
# Login
# -------------------------
USERS = {
    "officer": {"pwd": "123", "role": "Loan Officer"},
    "risk": {"pwd": "123", "role": "Risk Manager"},
    "ds": {"pwd": "123", "role": "Data Scientist"},
    "user": {"pwd": "123", "role": "End User"}
}

def login(username, password):
    user = USERS.get(username)
    if user and user["pwd"] == password:
        return f"✅ Đăng nhập thành công ({user['role']})", user["role"], gr.update(visible=False)
    return "❌ Sai username hoặc password", None, gr.update(visible=True)

def show_tabs(role):
    return (
        gr.update(visible=(role == "Loan Officer")),
        gr.update(visible=(role == "Risk Manager")),
        gr.update(visible=(role == "Data Scientist")),
        gr.update(visible=(role == "End User")),
    )

# -------------------------
# Mock scoring / utilities
# -------------------------
REQ_COLS = ["Age", "Income", "LoanAmount", "NumLoans", "DTI"]

def _score_row(age, income, loan_amt, num_loans, dti):
    z = 0.42 * (loan_amt / 100_000) + 0.22 * (num_loans / 5) + 0.28 * (dti / 60) \
        - 0.12 * (income / 200_000) - 0.06 * (age / 70)
    score = float(np.clip(0.5 + z, 0, 1))
    return score

def customer_query(phone_id):
    # Return mock profile, mock score and a bar plot for feature impacts
    profile = {
        "Họ tên": "Nguyễn Thị B",
        "Tuổi": 39,
        "Giới tính": "Nữ",
        "Thu nhập (năm, giả lập)": "120,000",
        "Số khoản vay hiện tại": 3,
        "DTI": "28%",
        "Số tiền vay đề nghị": "150,000"
    }
    score = _score_row(39, 120_000, 150_000, 3, 28)
    decision = (
        "Phê duyệt" if score <= 0.45 else
        "Phê duyệt có điều kiện" if score < 0.65 else
        "Từ chối"
    )

    # SHAP-like mock
    features = ["DTI", "LoanAmount", "NumLoans", "Income", "Age"]
    impacts = [0.30, 0.20, 0.18, -0.16, -0.12]

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars = ax.barh(features, impacts)
    ax.set_xlabel("Ảnh hưởng lên xác suất vỡ nợ (±)")
    ax.set_title("Top 5 yếu tố ảnh hưởng")
    ax.axvline(0, color="#777777", linewidth=0.6)
    plt.tight_layout()
    profile_text = "\n".join([f"{k}: {v}" for k, v in profile.items()])
    return profile_text, f"{score:.2f}", decision, fig

def risk_overview(_):
    # Pie distribution, trend line, histogram
    # Pie
    categories = ["Low", "Medium", "High"]
    values = [58, 30, 12]

    fig1, ax1 = plt.subplots(figsize=(3.6, 2.6))
    wedges, texts, autotexts = ax1.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, textprops={'fontsize':9})
    ax1.set_title("Phân bố rủi ro danh mục")

    # Trend
    months = pd.date_range(end=datetime.now(), periods=8, freq='M').strftime("%b %Y")
    trend = np.round(np.linspace(1.8, 3.6, len(months)) + np.random.rand(len(months))*0.2,2)
    fig2, ax2 = plt.subplots(figsize=(5.0, 2.6))
    ax2.plot(months, trend, marker='o', linewidth=2)
    ax2.set_title("Tỷ lệ vỡ nợ theo tháng")
    ax2.set_ylim(0, max(trend)*1.3)
    ax2.set_ylabel("%")

    # Histogram of scores
    scores = np.clip(np.random.beta(2,5,1000), 0, 1)
    fig3, ax3 = plt.subplots(figsize=(3.6, 2.6))
    ax3.hist(scores, bins=20)
    ax3.set_title("Phân bố Risk Score")
    ax3.set_xlabel("Risk Score")

    return fig1, fig2, fig3

def model_eval(_):
    # ROC-like, confusion matrix, metric table
    # ROC
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr) * 0.9 + 0.05
    fig1, ax1 = plt.subplots(figsize=(4.0, 3.0))
    ax1.plot(fpr, tpr, label='ROC (AUC=0.86)')
    ax1.plot([0,1],[0,1], 'k--', linewidth=0.6)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    # Confusion matrix
    cm = np.array([[820, 50],[70, 60]])
    fig2, ax2 = plt.subplots(figsize=(3.6, 2.8))
    im = ax2.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax2.text(j, i, f"{val}", ha='center', va='center', color='black', fontsize=10)
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Non-default","Default"]); ax2.set_yticklabels(["Actual Non-default","Actual Default"])
    ax2.set_title("Confusion Matrix")

    # Metrics table
    metrics = pd.DataFrame({
        "Metric": ["AUC","Accuracy","Precision","Recall","F1"],
        "Value": [0.86, 0.89, 0.83, 0.81, 0.82]
    })

    # Feature importance bar
    fi = pd.DataFrame({
        "Feature": ["DTI","LoanAmount","NumLoans","Income","Age"],
        "Importance": [0.33,0.26,0.18,0.13,0.10]
    })
    fig3, ax3 = plt.subplots(figsize=(4.0, 2.6))
    ax3.bar(fi["Feature"], fi["Importance"])
    ax3.set_title("Feature Importance")
    ax3.set_ylabel("Importance")

    return fig1, fig2, metrics, fig3

# -------------------------
# CSS & Theme
# -------------------------
# Light, modern palette with teal accents and warm accent color
CSS = """
:root{
  --bg:#f7fbfc;
  --card:#ffffff;
  --muted:#6b7280;
  --accent:#0e9aa7;    /* teal */
  --accent-2:#ff8a4c;  /* warm orange */
  --panel-border: rgba(16,24,40,0.06);
}
.gradio-container { background: var(--bg); color: #0f172a; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; padding: 16px; }
/* Card style for blocks */
.gr-box, .gr-panel, .gr-form, .gr-group { background: var(--card) !important; border: 1px solid var(--panel-border) !important; border-radius: 12px !important; padding: 14px !important; box-shadow: 0 6px 18px rgba(16,24,40,0.04); }
/* Headings */
.prose h1, .prose h2 { color: #0f172a !important; }
.prose p, .prose small { color: var(--muted) !important; }
/* Buttons */
button.primary { background: var(--accent) !important; color: white !important; border-radius: 10px !important; padding: 8px 12px !important; box-shadow: 0 6px 12px rgba(14,154,167,0.18); }
button.secondary { background: var(--accent-2) !important; color: white !important; border-radius: 10px !important; padding: 8px 12px !important; }
/* Inputs */
input, textarea, select { border-radius: 8px !important; border: 1px solid rgba(15,23,42,0.06) !important; padding: 8px !important; }
/* Small labels */
.label-wrap .label { color: var(--muted) !important; font-weight:600; }
/* Tables / Dataframe */
.dataframe thead th { background: transparent !important; color: #0f172a !important; font-weight:700; }
.dataframe tbody tr:nth-child(even) { background: #fbfcfd !important; }
/* Tabs */
.tabs { background: transparent !important; padding-bottom: 0; }
.tabitem.selected { border-bottom: 3px solid var(--accent) !important; }
"""

# -------------------------
# Build the Gradio app
# -------------------------
with gr.Blocks(title="Loan Default Prediction", css=CSS) as app:
    
    # Header
    with gr.Row(elem_id="header-row"):
        gr.Markdown(
            """
            # Loan Default Prediction
            """)
        with gr.Column(min_width=180):
            gr.Markdown("**Status:** Demo • no production data")
            gr.Button("Tải hướng dẫn (PDF)", variant="secondary")

    
    # =========== LOGIN BOX ============
    gr.Markdown("## 🔐 Đăng nhập hệ thống")
    login_box = gr.Group(visible=True)
    with login_box:
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Đăng nhập")
    
    # login_status = gr.Textbox(label="Trạng thái", interactive=False)
    login_status = gr.Markdown(label="Trạng thái")
    role_state = gr.State()
    
    gr.Markdown("---")

    with gr.Tabs():
        # -----------------------
        # Loan Officer Tab
        # -----------------------
        with gr.TabItem("Loan Officer", visible=False) as tab_officer:
            with gr.Row():
                # Left column: search + quick card
                with gr.Column():
                    gr.Markdown("#### Tìm hồ sơ / Tra cứu nhanh")
                    phone = gr.Textbox(label="Số điện thoại hoặc ApplicantID", placeholder="VD: 0912xxxxxx")
                    lookup_btn = gr.Button("Tra cứu", variant="primary")
                    profile_box = gr.Textbox(label="Thông tin hồ sơ", interactive=False, lines=8)

                    with gr.Row():
                        decision = gr.Radio(choices=["Phê duyệt", "Phê duyệt có điều kiện", "Từ chối"], label="Quyết định")

                    decision_feedback = gr.Markdown("*(Chưa có quyết định được chọn)*")
                    def on_decision_change(decision_choice):
                        return f"**Đã lưu: {decision_choice}**"
                    decision.change(fn=on_decision_change, inputs=[decision], outputs=[decision_feedback])

                # Right column: score + reasons
                with gr.Column():
                    gr.Markdown("#### Kết quả nhanh")
                    with gr.Row():
                        score_box = gr.Textbox(label="Risk Score (0..1)", interactive=False)
                        decision_box = gr.Textbox(label="Gợi ý quyết định", interactive=False)
                    gr.Markdown("**Giải thích ngắn**")
                    expl_plot = gr.Plot()
                    # Hook up the lookup
                    def _on_lookup(pid):
                        profile, score, decision, fig = customer_query(pid)
                        return profile, score, decision, fig
                    lookup_btn.click(fn=_on_lookup, inputs=phone, outputs=[profile_box, score_box, decision_box, expl_plot])

                    gr.Markdown("**Ghi chú của nhân viên**")
                    officer_note = gr.Textbox(label="Ghi chú (tùy chọn)", placeholder="Ghi chú ngắn cho hồ sơ")
                    save_note_btn = gr.Button("Lưu ghi chú", variant="secondary")
                    status_note = gr.Textbox(label="Trạng thái ghi chú", interactive=False)

                    def _save_note(note):
                        if not note:
                            return "Không có nội dung để lưu."
                        return "Đã lưu."
                    save_note_btn.click(fn=_save_note, inputs=officer_note, outputs=status_note)

            gr.Markdown("---")
            gr.Markdown("**Batch / Import**")
            with gr.Row():
                with gr.Column():
                    batch_file = gr.File(label="Tải CSV hồ sơ (Age,Income,LoanAmount,NumLoans,DTI)", file_types=[".csv"])
                    th1 = gr.Slider(0.1, 0.9, value=0.45, step=0.01, label="Ngưỡng 'Phê duyệt' nếu Risk <=")
                    th2 = gr.Slider(0.2, 0.95, value=0.65, step=0.01, label="Ngưỡng 'Có điều kiện' nếu Risk <")
                    run_batch = gr.Button("Chấm điểm batch & Sinh Proposal", variant="primary")
                    batch_status = gr.Textbox(label="Trạng thái batch", interactive=False)
                with gr.Column():
                    preview_table = gr.Dataframe(label="Xem trước", row_count=(5, 'dynamic'))
                    proposal_download = gr.File(label="Proposal CSV", interactive=False)

            def _run_batch(file, th_approve, th_conditional):
                if file is None:
                    return "Chưa chọn file.", pd.DataFrame(), None
                try:
                    df = pd.read_csv(file.name if hasattr(file, "name") else file)
                except Exception as e:
                    return f"Lỗi đọc CSV: {e}", pd.DataFrame(), None
                missing = [c for c in REQ_COLS if c not in df.columns]
                if missing:
                    return f"Thiếu cột: {', '.join(missing)}", df.head(5), None
                scores, decisions = [], []
                for _, r in df.iterrows():
                    s = _score_row(r["Age"], r["Income"], r["LoanAmount"], r["NumLoans"], r["DTI"])
                    dec = "Phê duyệt" if s <= th_approve else ("Phê duyệt có điều kiện" if s < th_conditional else "Từ chối")
                    scores.append(round(s, 4)); decisions.append(dec)
                out = df.copy()
                if "ApplicantID" not in out.columns:
                    out.insert(0, "ApplicantID", [f"CAND_{i+1:04d}" for i in range(len(out))])
                out["RiskScore"] = scores
                out["ProposedDecision"] = decisions
                # write csv temp
                tmp = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".csv")
                out.to_csv(tmp.name, index=False, encoding="utf-8")
                return f"Đã chấm {len(out)} hồ sơ.", out.head(5), tmp.name

            run_batch.click(fn=_run_batch, inputs=[batch_file, th1, th2], outputs=[batch_status, preview_table, proposal_download])

        # -----------------------
        # Risk Manager Tab
        # -----------------------
        with gr.TabItem("C-level /  Risk Manager", visible=False) as tab_risk:
            gr.Markdown("#### Risk Overview • Thống kê & Cảnh báo")
            # Controls
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Bộ lọc nhanh**")
                    date_range = gr.Slider(1, 24, value=12, label="Xem trong (tháng)", info="Chọn khoảng thời gian để xem xu hướng")
                    seg_select = gr.Dropdown(choices=["Toàn bộ", "Khu vực A", "Khu vực B", "Ad Source X"], value="Toàn bộ", label="Phân đoạn")
                    refresh_btn = gr.Button("Tải lại thống kê", variant="primary")
                with gr.Column():
                    key_kpis = gr.Markdown("**KPIs)**\n\n- Portfolio Size: **12,540**\n- Current Default Rate: **2.9%**\n- Avg Risk Score: **0.41**")
            # Charts
            with gr.Row():
                pie_plot = gr.Plot()
                trend_plot = gr.Plot()
                hist_plot = gr.Plot()

            def _refresh_kpi(_dr, seg):
                f1, f2, f3 = risk_overview(None)
                return f1, f2, f3
            refresh_btn.click(fn=_refresh_kpi, inputs=[date_range, seg_select], outputs=[pie_plot, trend_plot, hist_plot])

            gr.Markdown("---")
            gr.Markdown("**Stress Test / What-if**")
            with gr.Row():
                shock_slider = gr.Slider(0.0, 0.5, value=0.10, step=0.01, label="Shock lên default rate (+%)")
                run_shock = gr.Button("Chạy stress test", variant="secondary")
                shock_output = gr.Textbox(label="Kết quả", interactive=False)
            def _do_shock(shock):
                base = 2.9
                projected = round(base * (1 + shock), 2)
                return f"Tỷ lệ vỡ nợ dự phóng: {projected}%"
            run_shock.click(fn=_do_shock, inputs=shock_slider, outputs=shock_output)

        # -----------------------
        # Business Analyst / Data Scientist Tab
        # -----------------------
        with gr.TabItem("BA / Data Analyst / Data Scientist", visible=False) as tab_ds:
            gr.Markdown("#### Giám sát mô hình & phân tích đặc trưng")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Model metrics**")
                    eval_btn = gr.Button("Tải số liệu đánh giá", variant="primary")
                    roc_plot = gr.Plot()
                    cm_plot = gr.Plot()
                with gr.Column():
                    metric_table = gr.Dataframe(headers=["Metric","Value"], label="Metrics")
                    fi_plot = gr.Plot()
                    download_model = gr.Button("Tải model", variant="secondary")
                    model_file = gr.File(label="Model file", interactive=False)

            def _load_eval(_):
                f1, f2, metrics_df, f3 = model_eval(None)
                # create a small temp "model" file to simulate download
                tmp = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".bin")
                tmp.write(b"MODEL_BINARY")
                tmp.close()
                return f1, f2, metrics_df, f3, tmp.name

            eval_btn.click(fn=_load_eval, inputs=[gr.State()], outputs=[roc_plot, cm_plot, metric_table, fi_plot, model_file])

            gr.Markdown("---")
            gr.Markdown("**Notes / Observability**")
            drift_note = gr.Textbox(label="Drift warning", interactive=False, value="No significant feature drift detected in the last 30 days.")
            explain_note = gr.Textbox(label="Explainability tip", interactive=False, value="Use SHAP summary for cohort-level insights; check model behavior on low-income segment.")


        # -----------------------
        # End Users
        # -----------------------
        with gr.Tab("Người dùng cuối — Tra cứu kết quả", visible=False) as tab_user:
            gr.Markdown("### 🔍 Tra cứu kết quả vay vốn")
    
            with gr.Group():
                phone = gr.Textbox(label="Số điện thoại", placeholder="Nhập số điện thoại đã đăng ký", max_lines=1)
                send_otp_btn = gr.Button("Gửi mã OTP", variant="secondary")
                otp = gr.Textbox(label="Nhập OTP", placeholder="Nhập mã xác nhận gồm 6 chữ số")
                verify_btn = gr.Button("Xác thực & Tra cứu", variant="primary")
    
            result_status = gr.Textbox(label="Trạng thái tra cứu", interactive=False)
            user_info = gr.Dataframe(
                headers=["Họ tên", "Ngày duyệt", "Kết quả", "Lý do"],
                row_count=1, col_count=4, interactive=False, label="Kết quả tra cứu"
            )
    
            # --- Mock OTP backend ---
            import random
            otp_state = gr.State(value="")
    
            def _send_otp(phone):
                if not phone:
                    return "⚠️ Vui lòng nhập số điện thoại.", ""
                otp_code = f"{random.randint(100000, 999999)}"
                # (ở môi trường thật: gửi OTP qua SMS)
                return f"✅ OTP đã gửi đến {phone} (OTP: {otp_code})", otp_code
    
            send_otp_btn.click(fn=_send_otp, inputs=[phone], outputs=[result_status, otp_state])
    
            def _verify_otp(otp_input, otp_expected):
                if otp_input.strip() != otp_expected.strip():
                    return "❌ Mã OTP không đúng hoặc đã hết hạn.", pd.DataFrame()
                # Kết quả vay
                df = pd.DataFrame([{
                    "Họ tên": "Nguyễn Văn A",
                    "Duyệt lúc": "2025-11-03 15:20:00",
                    "Kết quả": "✅ Được phê duyệt có điều kiện",
                    "Lý do": "Thu nhập ổn định, DTI hợp lý. Cần bổ sung sao kê 3 tháng gần nhất."
                }])
                return "✅ Xác thực thành công!", df
    
            verify_btn.click(fn=_verify_otp, inputs=[otp, otp_state], outputs=[result_status, user_info])

        
        # -----------------------
        # Login
        # -----------------------
        login_btn.click(login, [username, password], [login_status, role_state, login_box])
        role_state.change(show_tabs, [role_state], [tab_officer, tab_risk, tab_ds, tab_user])


    # Footer
    gr.Markdown("---")
    gr.Markdown("© Demo UI — Loan Default Prediction • Designed for coursework / prototype")


if __name__ == '__main__':
    app.launch()