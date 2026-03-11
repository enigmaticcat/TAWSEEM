# TAWSEEM: TÀI LIỆU GIẢI THÍCH CHI TIẾT CÁC ĐẶC TRƯNG (FEATURE EXPLANATION)

Tài liệu này giải thích toàn bộ quy trình từ dữ liệu thô (Raw CSV) đến các đặc trưng tinh lọc được sử dụng để huấn luyện mô hình học sâu trong dự án TAWSEEM.

---

## 1. TỔNG QUAN QUY TRÌNH (PIPELINE)

Quy trình xử lý dữ liệu của TAWSEEM được chia làm 2 giai đoạn chính:

1. **Giai đoạn Tiền xử lý (Preprocessing):** Làm sạch dữ liệu từ CSV và tạo ra **72 đặc trưng thô** cho mỗi hàng (mỗi marker).
2. **Giai đoạn Trích xuất Ý nghĩa (Feature Engineering):** Biến đổi 72 số thô thành bộ **15 đặc trưng tinh gọn/marker** và **10 đặc trưng tổng hợp/mẫu**.

---

## 2. CHI TIẾT CÁC ĐẶC TRƯNG

### 2.1. Cấp độ 0: Dữ liệu Gốc (Input)

Dữ liệu được lấy từ 3 nhóm cột chính trong tệp PROVEDIt:

* **Height 1-10:** Chiều cao tín hiệu (RFU).
* **Allele 1-10:** Tên alen (số lần lặp lại).
* **Size 1-10:** Kích thước phân tử (bp).

### 2.2. Cấp độ 1: 72 Đặc trưng Thô (Preprocessing)

| Nhóm Đặc Trưng          | Số lượng | Ý nghĩa & Cách xử lý                                                                                                                    |
| :-------------------------- | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Peak Data (Value)** |      30      | Gồm 10 Allele, 10 Size, 10 Height. Thay thế "OL" bằng 0 để tính toán.                                                                 |
| **OL Indicator**      |      10      | Biến nhị phân (1 nếu giá trị gốc là "OL", 0 nếu không). Chốt lại thông tin ngoại lai.                                          |
| **Missing Indicator** |      30      | Biến nhị phân (1 nếu ô đó bị trống, 0 nếu có dữ liệu). Giúp mô hình phân biệt dữ liệu trống với giá trị 0 thực tế. |
| **Categorical**       |      2      | Gồm `Dye` (màu sắc) và `Marker` (tên gen) được mã hóa thành số.                                                              |
| **TỔNG CỘNG**       | **72** | **Đây là dữ liệu đầu vào cho bước tiếp theo.**                                                                              |

### 2.3. Cấp độ 2: 15 Đặc trưng Tinh lọc (Mỗi Marker)

Được tính toán để mô phỏng cách nhìn của chuyên gia pháp y:

1. **n_alleles:** Số lượng alen thực tế (không tính ô trống).
2. **h1, h2, h3:** Chiều cao của 3 đỉnh cao nhất (nhận diện người đóng góp chính).
3. **sum_h:** Tổng tín hiệu của marker.
4. **mean_h / std_h:** Độ mạnh trung bình và sự biến thiên của tín hiệu.
5. **h_ratio ($h1/h2$):** Chỉ số quan trọng nhất để tìm mẫu hỗn hợp (Lệch càng cao $\rightarrow$ càng dễ là hỗn hợp).
6. **h_range:** Khoảng cách giữa đỉnh cao nhất và thấp nhất.
7. **n_ol:** Số lượng alen ngoại lai trong marker.
8. **n_missing:** Đếm số lượng alen bị thiếu (dropout).
9. **stutter_ratio ($h_{min}/h1$):** Nhận diện nhiễu trong quá trình nhân bản ADN.
10. **snr_top2:** Tỷ lệ Tín hiệu/Nhiễu. Nếu thấp, chứng tỏ có rất nhiều đỉnh phụ (người phụ).
11. **log1p_h1 / log1p_sum_h:** Nén dải giá trị từ hàng vạn về hàng đơn vị để AI học ổn định.

### 2.4. Cấp độ 3: 10 Đặc trưng Tổng hợp (Toàn bộ Mẫu - Profile)

Được sử dụng cho mô hình MLP/XGBoost để có cái nhìn tổng quát:

* **MAC (Max Allele Count):** Giá trị alen lớn nhất trong toàn bộ 22 marker.
* **Markers 3+ / 5+:** Thống kê xem bao nhiêu vị trí gen có dấu hiệu hỗn hợp.
* **Total Peaks / Total Signal:** Tổng lượng thông tin ADN có trong mẫu.
* **Aggregate Stats:** Trung bình và độ lệch chuẩn của số alen và chiều cao đỉnh trên toàn bộ mẫu.
* 

1. **MAC (Max Allele Count)** : Số alen lớn nhất tìm thấy trong bất kỳ marker nào.
2. **Mean Allele Count** : Trung bình số alen trên mỗi marker.
3. **Std Allele Count** : Độ lệch chuẩn số alen giữa các marker.
4. **Count Markers 3+** : Đếm xem có bao nhiêu marker có từ 3 alen trở lên.
5. **Count Markers 5+** : Đếm xem có bao nhiêu marker có từ 5 alen trở lên.
6. **Total OL** : Tổng số lượng alen ngoại lai (Out-of-Ladder) của cả mẫu.
7. **Mean Max Height** : Trung bình chiều cao đỉnh cao nhất ($h1$) của các marker.
8. **Std Max Height** : Độ lệch chuẩn của chiều cao các đỉnh cao nhất.
9. **Total Peaks** : Tổng số lượng tất cả các alen phát hiện được trong mẫu.
10. **Total Signal** : Tổng toàn bộ chiều cao (RFU) của tất cả các đỉnh trong mẫu.

---

## 3. CÁC KHÁI NIỆM KỸ THUẬT QUAN TRỌNG

### 3.1. Tại sao dùng Log1p ($\ln(1+x)$)?

Trong pháp y, chiều cao đỉnh dao động từ 50 đến 30,000 RFU. Nếu đưa trực tiếp vào mạng Neural, các giá trị lớn sẽ làm cháy (saturation) các neuron. Phép Log giúp nén dữ liệu nhưng vẫn giữ nguyên tương quan tỉ lệ.

### 3.2. SNR_top2 (Signal-to-Noise Ratio)

Đặc trưng này coi 2 đỉnh cao nhất là "Tín hiệu" và phần còn lại là "Nhiễu". Trong mẫu 1 người, SNR sẽ cực kỳ lớn. Trong mẫu hỗn hợp, SNR sẽ nhỏ đi nhanh chóng vì có nhiều người đóng góp thêm.

---

## 4. CẤU TRÚC ĐẦU VÀO CHO MÔ HÌNH AI

* **Dạng phẳng (MLP/XGBoost):**
  $22 \text{ Marker} \times 15 \text{ Đặc trưng} + 10 \text{ Đặc trưng tổng hợp} = \mathbf{340}$ đặc trưng.
* **Dạng 2D (CNN):**
  Một ma trận kích thước **(22 hàng x 15 cột)**, mỗi hàng đại diện cho một vị trí gen.

---

*Tài liệu được tổng hợp bởi AI trợ lý cho dự án TAWSEEM.*
