# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** python3.10

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi: Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=648 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
     Record này nói về cả bức ảnh. Vì đây là tác vụ Phân loại ảnh (Image Classification), mô hình chỉ dự đoán một nhãn tổng quát duy nhất đại diện cho toàn bộ nội dung của bức ảnh chứ không xác định vị trí hay khoanh vùng từng vật thể riêng biệt.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp đi kèm với taxonomy/checkpoint (cụ thể ở đây là tập dữ liệu chuẩn ImageNet-1K), không phải do mô hình tự nghĩ ra.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    class_id giúp máy tính truy xuất chính xác và không bị nhầm lẫn giữa các mã định danh trong cơ sở dữ liệu.

    class_name giúp con người đọc và hiểu trực quan tên lớp.

    taxonomy_name xác định đúng bảng danh mục tham chiếu chuẩn (vì cùng một class_id hoặc class_name có thể có ý nghĩa khác nhau ở các bộ dữ liệu/taxonomy khác nhau).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần quy định rõ tiêu chí ưu tiên chọn một lớp duy nhất, ví dụ: ưu tiên chủ thể nằm ở trung tâm bức ảnh, chủ thể chiếm diện tích hiển thị lớn nhất, hoặc chủ thể quan trọng nhất theo mục tiêu bài toán.
6. Vì sao không được lấy `score` của model làm ground truth?
    Vì score chỉ là độ tin cậy mang tính xác suất thống kê của mô hình (mô hình hoàn toàn có thể dự đoán sai với độ tin cậy cao), trong khi Ground truth bắt buộc phải là đáp án chuẩn xác dựa trên thực tế và được kiểm duyệt bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Góc trên-trái ($x_1, y_1$) nằm tại tọa độ (385.33 px, 69.24 px).  
   Góc dưới-phải ($x_2, y_2$) nằm tại tọa độ (498.92 px, 348.92 px).   
   Khung hình chữ nhật này bao quanh người đầu bếp với chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Với ngưỡng 0.20: có 20 predictions (theo kết quả in ra tại ô so sánh trong notebook).

   Với ngưỡng 0.60: có 6 predictions.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Reviewer phải tốn nhiều thời gian và công sức hơn để kiểm tra, lọc bỏ các khung đoán sai (False Positives), xóa bớt các khung bị trùng lặp trên cùng một vật thể, đồng thời điều chỉnh lại vị trí các khung bị lệch.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   "Bounding box phải bao quanh sát viền ngoài của toàn bộ phần nhìn thấy của vật thể, không để thừa quá nhiều khoảng trống nền xung quanh và không cắt lẹm vào chi tiết của vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Vật thể: Các chiếc nồi/chảo đồng treo trên giá phía trên (bị chảo khác che khuất một phần) hoặc chiếc bàn bếp ở góc trái dưới (bị cắt mép bởi khung hình). 
   Cách xử lý: Đưa vào Quy tắc trong Guideline.   
   Vì sao: Vì việc vật thể bị che khuất một phần (occlusion) hoặc bị cắt ở rìa ảnh (truncation) là tình huống diễn ra rất phổ biến trong thực tế. Guideline cần đưa ra quy chuẩn rõ ràng (ví dụ: "Vẫn vẽ box nếu phần nhìn thấy lớn hơn 20%") để tất cả gán nhãn viên làm đồng nhất, không cần tốn thời gian escalation cho mỗi ảnh đơn lẻ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001; `class_name`=person; `score`=0.899318; số điểm polygon=`polygon_point_count`=348; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
  Biên dạng hình học chính xác (Exact Contour / Boundary): Xác định chính xác các đường cong, góc cạnh và viền của đối tượng thay vì chỉ gói gọn trong một hình chữ nhật bao quanh.
  Hình dạng / Thể tích mặt nạ thực tế (Shape & Mask Area): Bóc tách phần pixel thực sự thuộc về vật thể, giúp bỏ qua các vùng không gian trống hoặc ảnh nền (background) nằm bên trong bbox.
  Xử lý sự che khuất & cấu trúc phức tạp: Phản ánh được các vùng bị khuyết, các lỗ hổng bên trong đối tượng, hoặc phần cơ thể nghiêng/co gập (ví dụ: dáng đứng/ngồi của người person hay phần quai/lòng của bowl, potted plant).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
  Phân biệt các vật thể riêng lẻ cùng lớp trong một ảnh tĩnh (ví dụ: người A với người B), làm khóa chính (Primary Key) quản lý dữ liệu ở mức đối tượng (instance-level).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Bám sát biên pixel thực của vật thể; tăng mật độ điểm ở đoạn cong; chỉ vẽ phần nhìn thấy được (visible area) khi bị che khuất; không vẽ hình chữ nhật bao quanh thô.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Leo thang (Escalation) tới Trưởng nhóm/Quản lý để thống nhất chuẩn gán nhãn chung, tránh việc gán nhãn viên tự đoán làm mất tính nhất quán của dữ liệu.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn dạng chuỗi tên lớp duy nhất cho toàn bộ ảnh | Nhãn sai lớp (ví dụ: gán nhãn ảnh bếp thành "living room"); JSON thiếu trường class_name. | Gán 1 nhãn danh mục duy nhất đại diện cho toàn bộ nội dung của bức ảnh. | Kiểm tra nhãn được gán có đúng với nội dung chính của ảnh không. |
| Phát hiện vật thể | Tọa độ khung bao chữ nhật + Tên lớp (ví dụ: bbox_xyxy: [x1, y1, x2, y2], class_name: "person"). |Bbox vẽ thừa nền, thiếu đối tượng, bbox bị lệch; JSON lệch tọa độ hoặc score quá thấp.  |  Vẽ khung bao chữ nhật (bounding box) vừa khít từng đối tượng và gán nhãn tương ứng.|  Kiểm tra độ khít của khung bao, nhãn đúng của từng vật thể và các đối tượng bị bỏ sót.|
| Instance segmentation | Chuỗi tọa độ đa giác (polygon) ôm sát biên vật thể + ID + Tên lớp (ví dụ: polygon_xy, instance_id: "kitchen-001" |Polygon cắt lấn/rời rạc vật thể, gộp 2 đối tượng chung 1 ID; JSON bị sai polygon_point_count.  |Vẽ đường biên đa giác (polygon mask) chi tiết ôm sát ranh giới pixel của từng đối tượng riêng biệt.  | Kiểm tra độ chính xác của đường biên đa giác, phân biệt đúng các thể hiện (instance_id) và xử lý vùng che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu):Chỉ sử dụng các hình ảnh công khai có sẵn do bài lab cung cấp, tuyệt đối không tải lên Colab hay đẩy lên GitHub bất kỳ dữ liệu cá nhân hay dữ liệu nội bộ nào.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?:Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng lại ngay lập tức và báo cáo cho Lab Coach hoặc Giảng viên hướng dẫn để xin ý kiến xử lý (Escalation).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
