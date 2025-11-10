#!/bin/bash

# Test script for vLLM server
# Usage: ./test_vllm.sh [host] [port]

HOST="${1:-localhost}"
PORT="${2:-8000}"
API_KEY="thanglq12@vpbank"
API_URL="http://${HOST}:${PORT}/v1/chat/completions"

echo "Testing vLLM server at: $API_URL"
echo "=========================================="
echo ""

# Sample test request with Vietnamese compliance analysis
curl -X POST "$API_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
  "model": "models/Qwen2.5-1.5B-Instruct-lora_16bit",
  "messages": [
    {
      "role": "system",
      "content": "\nBạn là chuyên gia phân tích tuân thủ cuộc gọi trong lĩnh vực **thu hồi nợ tín dụng**.\n\nBối cảnh:\n- Tổng đài viên (Agent) gọi cho khách hàng (KH) để nhắc nợ hoặc xác nhận thanh toán.\n- Hệ thống cần phát hiện các **hành vi vi phạm quy trình, quy định hoặc đạo đức nghề nghiệp** của Agent.\n- Bạn chỉ được dựa trên **transcript hội thoại thực tế**, **không được suy đoán ngoài nội dung**.\n- Kết quả cần **ngắn gọn, nhất quán và có thể đọc tự động** bởi hệ thống kiểm định.\n"
    },
    {
      "role": "user",
      "content": "\nPhân tích vi phạm **\"Tiết lộ nguồn truy dấu thông tin KH/BT3\"**.\n\nBối cảnh:\n- KH hoặc BT3 có thể hỏi agent: \"Sao biết số điện thoại của tôi?\", \"Lấy số tôi ở đâu?\", \"Ai cho số tôi?\"\n- Nhiệm vụ của bạn: xác định xem **agent có tiết lộ nguồn cụ thể nơi lấy được số điện thoại hay không.**\n\nTiêu chí:\n- **Vi phạm** nếu agent tiết lộ **bất kỳ nguồn cụ thể nào** sau đây:\n  * Mạng xã hội: Facebook, Zalo, Instagram, TikTok, LinkedIn...\n  * Danh sách khuyến mãi, chương trình marketing\n  * Tên người cung cấp cụ thể (ví dụ: \"Anh Minh cho\", \"Chị Hoa cung cấp\")\n  * Nguồn dữ liệu bên ngoài: website, ứng dụng, đối tác thứ ba\n  * Bất kỳ thông tin cụ thể nào về cách thức thu thập số điện thoại\n\n- **Không vi phạm** nếu agent trả lời:\n  * \"Số này được đăng ký làm **người tham chiếu**\"\n  * \"Số này được đăng ký làm **người thân**\"\n  * \"Số này được đăng ký làm **người liên hệ khẩn cấp**\"\n  * \"Số này trong hồ sơ vay của khách hàng\"\n  * \"Đây là số đăng ký trong hợp đồng\"\n  * Hoặc agent **từ chối/né tránh** không trả lời cụ thể\n  * Hoặc agent nói chung chung: \"Đây là thông tin trong hệ thống\", \"Số được cung cấp khi đăng ký\"\n\nLưu ý:\n- Chỉ vi phạm khi agent **CHỦ ĐỘNG TIẾT LỘ NGUỒN CỤ THỂ**\n- Nếu agent nói \"không rõ\", \"không biết\", \"em không có thông tin này\" → **không vi phạm**\n- Nếu chỉ nói \"từ hồ sơ khách hàng\" mà không nói nguồn bên ngoài → **không vi phạm**\n\nCác ví dụ:\n- ❌ Vi phạm: \"Số của anh lấy từ Facebook ạ\"\n- ❌ Vi phạm: \"Chị Lan ở phòng marketing cho số của anh\"\n- ❌ Vi phạm: \"Đây là số trong danh sách khuyến mãi\"\n- ✅ Không vi phạm: \"Đây là số người tham chiếu trong hồ sơ vay\"\n- ✅ Không vi phạm: \"Số này anh đăng ký làm người liên hệ khẩn cấp\"\n- ✅ Không vi phạm: \"Em không rõ nguồn cụ thể ạ\"\n\nTrả về định dạng XML (không thêm bất cứ thông tin thừa nào):\n<result>\n<violation>true/false</violation>\n<reason>Lý do ngắn gọn tại sao vi phạm hoặc không vi phạm</reason>\n<speech>Nội dung câu nói vi phạm (nếu có)</speech>\n<timespan>Thời điểm bắt đầu - kết thúc (giây)</timespan>\n</result>\n\nLưu ý: Nếu không vi phạm, để trống tag <speech>, <timespan>\n\nĐây là nội dung cuộc gọi (mỗi turn có start_time và end_time):\n[472.87s - 473.31s] CUSTOMER: được rồi\n[473.22s - 474.43s] AGENT: công ty của chị chẩm lương\n[474.82s - 476.41s] AGENT: như thế nào chậm bao nhiêu tháng\n"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}'

echo ""
echo ""
echo "=========================================="
echo "Test completed!"
