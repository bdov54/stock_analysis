from typing import Dict, List, Optional


def _format_metrics(metrics: Dict) -> str:
    if not metrics:
        return "- Chưa có dữ liệu chỉ số"

    lines = []
    for k, v in metrics.items():
        if v is None or v == "":
            continue
        lines.append(f"- {k}: {v}")

    return "\n".join(lines) if lines else "- Chưa có dữ liệu chỉ số"


def _format_list(items: Optional[List[str]], empty_text: str = "Chưa có dữ liệu") -> str:
    if not items:
        return f"- {empty_text}"
    return "\n".join([f"- {x}" for x in items])


def build_stock_prompt(
    stock_code: str,
    company_name: str,
    industry_name: str,
    metrics: Dict,
    macro_summary: str = "",
    macro_positives: Optional[List[str]] = None,
    macro_risks: Optional[List[str]] = None,
) -> str:
    metric_block = _format_metrics(metrics)
    positives_block = _format_list(macro_positives, "Không có điểm hỗ trợ nổi bật")
    risks_block = _format_list(macro_risks, "Không có rủi ro nổi bật")

    return f"""
Bạn là trợ lý phân tích đầu tư, viết bằng tiếng Việt dễ hiểu cho người không chuyên.

NGUYÊN TẮC BẮT BUỘC:
- Chỉ sử dụng dữ liệu được cung cấp dưới đây.
- Không bịa số liệu, không suy diễn quá mức.
- Không đưa ra khuyến nghị mua/bán chắc chắn.
- Nếu dữ liệu chưa đủ, phải nói rõ là dữ liệu chưa đủ.
- Văn phong rõ ràng, ngắn gọn, dễ hiểu.

THÔNG TIN CỔ PHIẾU
- Mã cổ phiếu: {stock_code}
- Tên công ty: {company_name}
- Lĩnh vực hoạt động: {industry_name}

CHỈ SỐ TÀI CHÍNH
{metric_block}

BỐI CẢNH VĨ MÔ
{macro_summary if macro_summary else "Chưa có mô tả vĩ mô"}

ĐIỂM HỖ TRỢ VĨ MÔ / NGÀNH
{positives_block}

RỦI RO VĨ MÔ / NGÀNH
{risks_block}

Hãy viết đúng theo cấu trúc sau:

1. Tóm tắt ngắn
- Viết 2-3 câu ngắn giải thích doanh nghiệp này là công ty như thế nào và vì sao xuất hiện trong danh sách.

2. Điểm mạnh nổi bật
- Liệt kê 3 ý ngắn, bám sát dữ liệu đầu vào.

3. Rủi ro cần chú ý
- Liệt kê 2-3 ý ngắn, bám sát dữ liệu đầu vào và bối cảnh vĩ mô.

4. Kết luận dễ hiểu
- Viết 2 câu ngắn cho người không chuyên, theo hướng trung lập, không hô hào đầu tư.
""".strip()


def build_portfolio_prompt(
    portfolio_rows: List[Dict],
    macro_summary: str = "",
    macro_positives: Optional[List[str]] = None,
    macro_risks: Optional[List[str]] = None,
) -> str:
    if portfolio_rows:
        stock_lines = []
        for i, row in enumerate(portfolio_rows, start=1):
            stock_lines.append(
                f"{i}. Mã: {row.get('CompID', '')} | "
                f"Tên: {row.get('Company Common Name', '')} | "
                f"Lĩnh vực: {row.get('TRBC Industry Name', row.get('GICS Sub-Industry Name', row.get('industry_bucket', '')))}"
            )
        portfolio_block = "\n".join(stock_lines)
    else:
        portfolio_block = "Chưa có dữ liệu danh mục"

    positives_block = _format_list(macro_positives, "Không có điểm hỗ trợ nổi bật")
    risks_block = _format_list(macro_risks, "Không có rủi ro nổi bật")

    return f"""
Bạn là trợ lý phân tích đầu tư, viết bằng tiếng Việt dễ hiểu cho người không chuyên.

NGUYÊN TẮC BẮT BUỘC:
- Chỉ sử dụng dữ liệu được cung cấp dưới đây.
- Không bịa số liệu.
- Không đưa ra khuyến nghị mua/bán chắc chắn.
- Nếu dữ liệu chưa đủ, phải nói rõ là dữ liệu chưa đủ.
- Văn phong ngắn gọn, rõ ràng, dễ đọc.

DANH MỤC HIỆN TẠI
{portfolio_block}

BỐI CẢNH VĨ MÔ
{macro_summary if macro_summary else "Chưa có mô tả vĩ mô"}

ĐIỂM HỖ TRỢ VĨ MÔ / NGÀNH
{positives_block}

RỦI RO VĨ MÔ / NGÀNH
{risks_block}

Hãy viết đúng theo cấu trúc sau:

1. Tóm tắt danh mục
- Viết 2-3 câu ngắn mô tả danh mục này đang tập trung vào kiểu doanh nghiệp nào.

2. Điểm tích cực của danh mục
- Liệt kê 3 ý ngắn.

3. Rủi ro cần chú ý
- Liệt kê 3 ý ngắn.

4. Kết luận dễ hiểu
- Viết 2 câu ngắn, trung lập, phù hợp với người không chuyên.
""".strip()