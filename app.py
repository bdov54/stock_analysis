import streamlit as st
import pandas as pd

from config import AppConfig
from insight_engine import get_macro_insights
from pipeline import run_pipeline
from reporting import plot_stock_detail
from ai_commentary import generate_ai_commentary
from prompt_builder import build_stock_prompt, build_portfolio_prompt


st.set_page_config(page_title="ATHEX Stock Selector", layout="wide")
st.title("ATHEX Quality Growth Stock Selector")
st.caption("Công cụ sàng lọc cổ phiếu tăng trưởng chất lượng trên thị trường ATHEX.")


# =========================
# Helpers
# =========================
def get_display_industry_col(df: pd.DataFrame):
    for c in ["TRBC Industry Name", "GICS Sub-Industry Name", "industry_bucket"]:
        if c in df.columns:
            return c
    return None


def pick_display_cols(df: pd.DataFrame):
    cols = []
    if "CompID" in df.columns:
        cols.append("CompID")
    if "Company Common Name" in df.columns:
        cols.append("Company Common Name")

    industry_col = get_display_industry_col(df)
    if industry_col is not None:
        cols.append(industry_col)

    return cols


def build_ai_metrics_for_stock(round2_df: pd.DataFrame, selected: str):
    metrics = {}
    if len(round2_df) == 0 or "CompID" not in round2_df.columns:
        return metrics

    metric_source = round2_df[round2_df["CompID"] == selected]
    if len(metric_source) == 0:
        return metrics

    metric_row = metric_source.iloc[0]
    for c in [
        "MED_ROE",
        "MED_ROA",
        "MED_ROIC",
        "MED_EBIT_margin",
        "MED_REV_CAGR_3Y",
        "MED_EPS_CAGR_3Y",
        "MED_CFO_NI",
        "MED_D_E",
        "MED_NetDebt_EBITDA",
        "MED_Current_Ratio",
    ]:
        if c in metric_row.index and pd.notna(metric_row[c]):
            try:
                metrics[c] = round(float(metric_row[c]), 4)
            except Exception:
                pass
    return metrics


def render_usage_page():
    st.subheader("Hướng dẫn sử dụng")
    st.write(
        "Trang này giải thích cách dùng công cụ theo cách đơn giản nhất. "
        "Bạn không cần biết về lập trình hay tài chính chuyên sâu vẫn có thể sử dụng."
    )

    st.markdown("### Công cụ này dùng để làm gì?")
    st.write(
        "Công cụ sẽ đọc dữ liệu tài chính của các công ty trong file `data/Greece.xlsx`, "
        "sau đó tự động lọc ra những công ty có nền tảng tài chính tốt hơn, "
        "chấm điểm và gợi ý một danh mục cổ phiếu."
    )

    st.markdown("### Bạn cần làm gì?")
    st.write("Bạn chỉ cần 3 bước:")
    st.write("1. Chỉnh các điều kiện lọc cơ bản ở thanh bên trái nếu muốn.")
    st.write("2. Bấm nút **Run pipeline**.")
    st.write("3. Xem kết quả ở các tab bên dưới.")

    st.markdown("### Ý nghĩa các điều kiện bên trái")
    st.write("**Số cổ phiếu trong danh mục**: số mã bạn muốn công cụ đề xuất.")
    st.write("**Min coverage**: mức độ dữ liệu đầy đủ tối thiểu của công ty.")
    st.write("**Min CFO/NI**: mức tối thiểu của dòng tiền hoạt động so với lợi nhuận.")
    st.write("**Max D/E**: mức nợ trên vốn chủ tối đa cho phép.")
    st.write("**Max NetDebt/EBITDA**: mức nợ ròng so với EBITDA tối đa.")
    st.write("**Min ROE**: mức sinh lời trên vốn chủ tối thiểu.")

    st.markdown("### Kết quả được hiểu như thế nào?")
    st.write(
        "- **Danh mục**: nhóm cổ phiếu được đề xuất sau khi hệ thống lọc và đa dạng hóa.\n"
        "- **Tổng quan**: cho biết dữ liệu đã đi qua những bước nào và còn lại bao nhiêu công ty.\n"
        "- **Phân tích cổ phiếu**: xem chi tiết một công ty cụ thể qua biểu đồ.\n"
        "- **Vĩ mô & ngành**: xem bối cảnh kinh tế và các yếu tố ngành."
    )

    st.markdown("### Nếu không ra kết quả thì sao?")
    st.write(
        "Có thể do điều kiện lọc đang quá chặt. "
        "Bạn hãy giảm bớt mức khó của các điều kiện, ví dụ giảm Min CFO/NI hoặc tăng Max D/E."
    )

    st.markdown("### Lưu ý quan trọng")
    st.write(
        "Danh mục được gợi ý để hỗ trợ phân tích, không phải cam kết chắc chắn sinh lời. "
        "Bạn vẫn nên xem thêm phần chi tiết cổ phiếu và bối cảnh ngành trước khi ra quyết định."
    )


# =========================
# Session state
# =========================
if "results" not in st.session_state:
    st.session_state["results"] = None

if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = None

if "stock_ai_commentary" not in st.session_state:
    st.session_state["stock_ai_commentary"] = {}

if "portfolio_ai_commentary" not in st.session_state:
    st.session_state["portfolio_ai_commentary"] = None


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Thiết lập")
    st.markdown("**Nguồn dữ liệu:** `data/Greece.xlsx`")
    st.markdown("Bạn chỉ cần chỉnh các điều kiện lọc cơ bản, hệ thống sẽ tự chạy phần kỹ thuật ở bên trong.")

    portfolio_size = st.slider("Số cổ phiếu trong danh mục", 3, 15, 7)

    st.subheader("Hard filter")
    min_coverage = st.slider("Min coverage", 0.0, 1.0, 0.60, 0.05)
    min_cfo_ni = st.number_input("Min CFO/NI", value=0.50, step=0.05)
    max_de = st.number_input("Max D/E", value=2.50, step=0.10)
    max_netdebt = st.number_input("Max NetDebt/EBITDA", value=4.00, step=0.10)
    min_roe = st.number_input("Min ROE", value=0.00, step=0.01)

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)


# =========================
# Run pipeline
# =========================
if run_btn:
    cfg = AppConfig(
        file_path="data/Greece.xlsx",
        mode="data-driven",
        portfolio_size=int(portfolio_size),
        hard_filter_rules={
            "min_coverage": float(min_coverage),
            "min_cfo_ni": float(min_cfo_ni),
            "max_de_ratio": float(max_de),
            "max_netdebt_ebitda": float(max_netdebt),
            "min_roe": float(min_roe),
            "require_positive_equity": True,
        },
    )

    with st.spinner("Đang chạy pipeline..."):
        st.session_state["results"] = run_pipeline(cfg)
        st.session_state["selected_stock"] = None
        st.session_state["stock_ai_commentary"] = {}
        st.session_state["portfolio_ai_commentary"] = None


results = st.session_state["results"]

if results is None:
    st.info("Dữ liệu đang lấy từ `data/Greece.xlsx`. Hãy bấm **Run pipeline** để bắt đầu.")
    render_usage_page()
    st.stop()


# =========================
# Read pipeline outputs
# =========================
meta = results.get("meta", {})
yearly_df = results.get("yearly_df", pd.DataFrame())
feature_df = results.get("feature_df", pd.DataFrame())
round1_df = results.get("round1_df", pd.DataFrame())
round2_df = results.get("round2_df", pd.DataFrame())
ranked_df = results.get("ranked_df", pd.DataFrame())
portfolio_df = results.get("portfolio_df", pd.DataFrame())
cluster_artifacts = results.get("cluster_artifacts", {})

# Không hiện cluster / score ra ngoài UI
portfolio_view = portfolio_df.drop(columns=["cluster", "TOTAL_SCORE", "PASS_COUNT", "ROUND2_PASS"], errors="ignore").copy()
ranked_view = ranked_df.drop(columns=["cluster", "TOTAL_SCORE", "PASS_COUNT", "ROUND2_PASS"], errors="ignore").copy()

portfolio_cols = pick_display_cols(portfolio_view)
rank_cols = pick_display_cols(ranked_view)

# Báo cáo luồng xử lý đơn giản, không show score
stage_report = pd.DataFrame(
    [
        {"Bước": "Load dữ liệu", "Số lượng": meta.get("companies", None)},
        {"Bước": "Feature engineering", "Số lượng": len(feature_df)},
        {"Bước": "Qua vòng 1 (hard filter)", "Số lượng": len(round1_df)},
        {"Bước": "Qua vòng 2 (scoring)", "Số lượng": len(round2_df)},
        {"Bước": "Danh mục cuối cùng", "Số lượng": len(portfolio_df)},
    ]
)


# =========================
# Tabs
# =========================
tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Cách sử dụng", "Danh mục", "Tổng quan", "Phân tích cổ phiếu", "Vĩ mô & ngành"]
)

with tab0:
    render_usage_page()

with tab1:
    st.subheader("Danh mục gợi ý")
    if len(portfolio_view) == 0:
        st.warning("Danh mục hiện đang rỗng.")
    else:
        if portfolio_cols:
            st.dataframe(portfolio_view[portfolio_cols], use_container_width=True)
        else:
            st.dataframe(portfolio_view, use_container_width=True)

        # AI commentary for portfolio
        if st.button("Tạo nhận xét AI cho danh mục"):
            macro = get_macro_insights()
            portfolio_rows = portfolio_view.to_dict(orient="records")
            with st.spinner("AI đang tạo nhận xét cho danh mục..."):
                prompt = build_portfolio_prompt(
                    portfolio_rows=portfolio_rows,
                    macro_summary=macro.get("summary", ""),
                    macro_positives=macro.get("positives", []),
                    macro_risks=macro.get("risks", []),
                )
                st.session_state["portfolio_ai_commentary"] = generate_ai_commentary(prompt)

        if st.session_state["portfolio_ai_commentary"]:
            st.markdown("### Nhận xét AI cho danh mục")
            st.write(st.session_state["portfolio_ai_commentary"])

with tab2:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số công ty", meta.get("companies", "-"))
    c2.metric("Số dòng company-year", meta.get("rows", "-"))
    c3.metric("Số mã qua vòng 1", len(round1_df))

    c4, c5, c6 = st.columns(3)
    c4.metric("Số mã qua vòng 2", len(round2_df))
    c5.metric("Số mã trong danh mục", len(portfolio_df))
    c6.metric("Silhouette", round(cluster_artifacts.get("silhouette", 0), 4) if cluster_artifacts else "-")

    st.subheader("Luồng xử lý")
    st.dataframe(stage_report, use_container_width=True)

    st.subheader("Danh sách công ty qua sàng lọc")
    if len(ranked_view) > 0:
        if rank_cols:
            st.dataframe(ranked_view[rank_cols].head(30), use_container_width=True)
        else:
            st.dataframe(ranked_view.head(30), use_container_width=True)
    else:
        st.info("Chưa có dữ liệu.")

with tab3:
    st.subheader("Phân tích chi tiết cổ phiếu")

    stock_source_df = round2_df if len(round2_df) > 0 else ranked_df
    stock_list = stock_source_df["CompID"].dropna().tolist() if "CompID" in stock_source_df.columns else []

    if stock_list:
        selected = st.selectbox(
            "Chọn cổ phiếu",
            stock_list,
            key="selected_stock",
        )

        detail_rows = ranked_view[ranked_view["CompID"] == selected].copy()

        if len(detail_rows) > 0:
            detail_cols = pick_display_cols(detail_rows)

            if detail_cols:
                st.dataframe(detail_rows[detail_cols], use_container_width=True)
            else:
                st.dataframe(detail_rows, use_container_width=True)

            if len(yearly_df) > 0:
                fig = plot_stock_detail(yearly_df, selected)
                st.pyplot(fig)

            # AI commentary for selected stock
            row = detail_rows.iloc[0]
            company_name = row["Company Common Name"] if "Company Common Name" in row.index else selected

            industry_name = ""
            for c in ["TRBC Industry Name", "GICS Sub-Industry Name", "industry_bucket"]:
                if c in row.index and pd.notna(row[c]):
                    industry_name = row[c]
                    break

            metrics = build_ai_metrics_for_stock(round2_df, selected)
            macro = get_macro_insights()

            if st.button("Tạo nhận xét AI cho cổ phiếu này"):
                with st.spinner("AI đang tạo nhận xét..."):
                    prompt = build_stock_prompt(
                        stock_code=selected,
                        company_name=company_name,
                        industry_name=industry_name,
                        metrics=metrics,
                        macro_summary=macro.get("summary", ""),
                        macro_positives=macro.get("positives", []),
                        macro_risks=macro.get("risks", []),
                    )
                    st.session_state["stock_ai_commentary"][selected] = generate_ai_commentary(prompt)

            if selected in st.session_state["stock_ai_commentary"]:
                st.markdown("### Nhận xét AI")
                st.write(st.session_state["stock_ai_commentary"][selected])
        else:
            st.info("Không tìm thấy dữ liệu chi tiết cho cổ phiếu này.")
    else:
        st.info("Chưa có dữ liệu cổ phiếu để hiển thị.")

with tab4:
    macro = get_macro_insights()

    st.subheader(macro.get("title", "Bối cảnh vĩ mô"))
    st.write(macro.get("summary", ""))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Điểm hỗ trợ**")
        for item in macro.get("positives", []):
            st.write(f"- {item}")

    with c2:
        st.markdown("**Rủi ro chính**")
        for item in macro.get("risks", []):
            st.write(f"- {item}")

    industry_col = get_display_industry_col(portfolio_view)
    if industry_col is not None:
        st.subheader("Lĩnh vực hoạt động trong danh mục")
        show_cols = ["CompID"]
        if "Company Common Name" in portfolio_view.columns:
            show_cols.append("Company Common Name")
        show_cols.append(industry_col)
        st.dataframe(portfolio_view[show_cols], use_container_width=True)