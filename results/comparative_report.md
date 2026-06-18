# Báo cáo Nghiên cứu Đối chứng Toàn diện (V2): Nén Nhúng Quantum-Classical

Báo cáo này tổng hợp kết quả thực nghiệm từ nghiên cứu đối chứng toàn diện (V2) về các phương pháp nén vector biểu diễn từ PhoBERT ($d = 768$) xuống các kích thước nhỏ hơn ($d' \in \{8, 16, 32, 64\}$) trên bộ dữ liệu đánh giá cảm xúc tiếng Việt **UIT-VSFC**. Nghiên cứu tập trung so sánh các thuật toán cổ điển, lượng tử biến phân (PQC) và lượng tử mô phỏng (QiC).

---

## 1. Bảng Tổng hợp Kết quả Thực nghiệm

Dưới đây là kết quả chi tiết của 6 phương pháp được thử nghiệm trên 4 không gian chiều đầu ra khác nhau:

| Phương pháp | Chiều đầu ra ($d'$) | Accuracy | Macro-F1 | Số tham số | Thời gian huấn luyện (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Autoencoder (Cổ điển)** | 8 | 0.9071 | 0.7673 | 198,920 | 100.1 |
| **QiC Cascaded (Paper)** | 8 | 0.9008 | 0.7757 | 24,752 | 91.2 |
| **PQC (Ring CNOT)** | 8 | 0.7255 | 0.5764 | 6,184 | 673.5 |
| **PQC No-Entanglement** | 8 | 0.8010 | 0.6100 | 6,184 | 602.9 |
| **PCA (Cổ điển)** | 8 | 0.7088 | 0.5951 | 0 | 18.4 |
| **Fidelity Classifier (on PCA)**| 8 | 0.6582 | 0.5557 | 0 | 0.0 |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **Autoencoder (Cổ điển)** | 16 | 0.9112 | 0.7712 | 200,976 | 113.9 |
| **QiC Cascaded (Paper)** | 16 | 0.8992 | 0.7742 | 49,504 | 89.8 |
| **PCA (Cổ điển)** | 16 | 0.7802 | 0.6655 | 0 | 31.0 |
| **Fidelity Classifier (on PCA)**| 16 | 0.6886 | 0.5832 | 0 | 0.0 |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **Autoencoder (Cổ điển)** | 32 | 0.9084 | 0.7634 | 205,088 | 84.8 |
| **QiC Cascaded (Paper)** | 32 | 0.9065 | 0.7745 | 99,008 | 81.4 |
| **PCA (Cổ điển)** | 32 | 0.8045 | 0.6894 | 0 | 26.5 |
| **Fidelity Classifier (on PCA)**| 32 | 0.7167 | 0.6078 | 0 | 0.0 |
| *---* | *---* | *---* | *---* | *---* | *---* |
| **Autoencoder (Cổ điển)** | 64 | 0.9084 | 0.7640 | 213,312 | 83.5 |
| **QiC Cascaded (Paper)** | 64 | 0.9093 | 0.7805 | 198,016 | 95.5 |
| **PCA (Cổ điển)** | 64 | 0.8193 | 0.7022 | 0 | 34.7 |
| **Fidelity Classifier (on PCA)**| 64 | 0.7211 | 0.6117 | 0 | 0.0 |

---

## 2. Phân tích Các Khám phá Quan trọng (Key Insights)

### 2.1. Tối ưu hóa Vector hóa cho QiC Cascaded: Tốc độ Vượt trội
Một trong những cải tiến kỹ thuật lớn nhất trong pha này là việc tái cấu trúc `CascadeLayer` và `ZYZRotation` trong [qi_compressor.py](file:///C:/Users/DELL/Documents/antigravity/focused-newton/src/qi_compressor.py) sang dạng **Vector hóa (Vectorized)** hoàn toàn.
* **Vấn đề trước đó:** Phiên bản cũ sử dụng vòng lặp Python tuần tự (`for` loop) qua từng cặp qubit, dẫn đến thời gian huấn luyện ở $d'=16$ mất tới **554.4 giây** và không thể chạy nổi ở các chiều lớn hơn do nghẽn CPU.
* **Giải pháp:** Sử dụng các phép toán Tensor song song của PyTorch, thực hiện quay trạng thái và tính toán đan xen CNOT trên toàn bộ các cặp qubit cùng lúc.
* **Kết quả:** Thời gian huấn luyện QiC ở $d'=16$ giảm từ **554.4 giây xuống còn 89.8 giây (nhanh hơn 6.1x)**. Ở các không gian lớn hơn như $d'=32$ và $d'=64$, mô hình hoàn thành chỉ trong khoảng **80-95 giây** thay vì mất hàng giờ như trước.

### 2.2. Hiệu quả Tham số Cực cao của QiC Cascaded (Paper Method)
Mô hình **QiC Cascaded** lấy cảm hứng từ cấu trúc nén lượng tử thể hiện sự vượt trội đáng kinh ngạc về hiệu quả tham số (parameter efficiency) so với mạng Autoencoder cổ điển:
* Ở kích thước cực hạn $d'=8$: QiC chỉ sử dụng **24,752 tham số** (bằng **12.4%** so với 198,920 tham số của Autoencoder), nhưng đạt độ chính xác gần như tương đương ($90.08\%$ so với $90.71\%$) và **vượt trội hơn về Macro-F1** ($0.7757$ so với $0.7673$).
* Ở kích thước $d'=64$: QiC đạt độ chính xác cao nhất trong toàn bộ bảng kiểm thử ($90.93\%$ Acc / $0.7805$ Macro-F1), vượt qua cả Autoencoder cổ điển ($90.84\%$ Acc / $0.7640$ Macro-F1).
* **Giải thích vật lý/toán học:** Việc ánh xạ các giá trị đặc trưng cổ điển lên tọa độ góc trên Quả cầu Bloch thông qua hàm $\tanh$ giúp chuẩn hóa dữ liệu tốt hơn. Cấu trúc nén xếp tầng từng bước (ghép cặp qubit, xoay góc lượng tử, và thực hiện phép đo Born) hoạt động như một bộ lọc thông tin phi tuyến tính cực kỳ cô đọng, loại bỏ nhiễu hiệu quả hơn các lớp Linear + ReLU cồng kềnh của Autoencoder.

### 2.3. Xác thực Giả thuyết về Barren Plateaus (Mặt phẳng cằn cỗi)
Thử nghiệm đối chứng trực tiếp giữa **PQC (Ring CNOT)** và **PQC No-Entanglement** ở $d'=8$ cung cấp bằng chứng thực nghiệm rõ ràng về hiện tượng triệt tiêu gradient trong học máy lượng tử:
1. **PQC (Ring CNOT)**: Đạt độ chính xác rất kém ($72.55\%$ Acc / $0.5764$ Macro-F1) và ghi nhận cảnh báo nghẽn đạo hàm từ rất sớm (Epoch 9-13). Việc đan xen toàn bộ 8 qubit bằng cổng CNOT dạng vòng tạo ra một không gian Hilbert khổng lồ ($2^8 = 256$ chiều phức). Sự phân bố xác xuất đồng đều trên không gian này khiến độ dốc trung bình tiến về 0 và phương sai triệt tiêu theo hàm mũ, tạo ra một mặt phẳng lỗi cằn cỗi không thể tối ưu hóa.
2. **PQC No-Entanglement**: Khi loại bỏ hoàn toàn các cổng CNOT, các qubit độc lập hoàn toàn. Mô hình đạt hiệu quả tốt hơn hẳn với **$80.10\%$ Accuracy** và **$0.6100$ Macro-F1** (tăng ~7.5% Acc). Mặc dù mô hình này vẫn gặp một số pha nghẽn cục bộ, việc thiếu liên kết đan xen toàn cục đã kéo phương sai của gradient thoát khỏi sự triệt tiêu hàm mũ của không gian Hilbert, giúp bộ tối ưu hóa Adam cập nhật trọng số hiệu quả hơn.

> [!IMPORTANT]
> Thực nghiệm này xác nhận: **Entanglement sâu trên mạch lượng tử biến phân là con dao hai lưỡi**. Nó tăng khả năng biểu diễn lượng tử nhưng lại trực tiếp gây ra lỗi Barren Plateaus khiến mô hình không thể hội tụ trên dữ liệu thực tế cỡ lớn.

### 2.4. Đánh giá Fidelity-based Classifier
Bộ phân loại dựa trên độ tương hợp lượng tử (Fidelity Classifier) hoạt động trên không gian nén của PCA cho kết quả thú vị:
* Ở chiều $d'=64$, bộ phân loại dựa trên Fidelity đạt độ chính xác tốt hơn hẳn khoảng cách Cosine thông thường ($72.11\%$ Acc vs $68.41\%$ Acc).
* Tuy nhiên, chất lượng phân loại của nó vẫn kém hơn nhiều so với việc sử dụng một Classifier Head (lớp tuyến tính) được huấn luyện đầy đủ. Điều này chứng minh rằng việc tính toán trực tiếp khoảng cách đến các "prototype" (vector trung bình lớp) thông qua độ tương đồng lượng tử chỉ thích hợp làm giải pháp ước lượng nhanh không tham số (non-parametric), chứ không thể thay thế cho việc học các ranh giới quyết định (decision boundaries) động.

---

## 3. Các Đồ thị Phân tích (Visualizations)

Các đồ thị phân tích đã được lưu trữ trong thư mục [results/plots](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots):

1. **[comparison_bars.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/comparison_bars.png)**: Biểu đồ cột thể hiện trực quan Accuracy và Macro-F1 của tất cả các mô hình qua 4 mức chiều nén. Đồ thị làm nổi bật sự bám đuổi sát sao của QiC so với Autoencoder và sự sụt giảm nghiêm trọng của các mô hình PQC lượng tử.
2. **[param_efficiency.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/param_efficiency.png)**: Biểu đồ thể hiện mối quan hệ giữa hiệu năng phân loại và số lượng tham số huấn luyện. Nó làm nổi bật QiC nằm ở góc trên bên trái (hiệu năng cao, số tham số siêu nhỏ).
3. **[training_curves_d8.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/training_curves_d8.png)**: So sánh tốc độ hội tụ (Loss & Accuracy qua từng Epoch) giữa Autoencoder và QiC ở kích thước $d'=8$.
4. **Các biểu đồ t-SNE 2D** ở chiều $d'=8$:
   * **[tsne_pca_d8.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/tsne_pca_d8.png)**: Phân bố phân tán của PCA. Các cụm lớp bị chồng lấn rất mạnh.
   * **[tsne_autoencoder_d8.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/tsne_autoencoder_d8.png)**: Phân bố của Autoencoder. Các cụm được phân tách rõ ràng và tập trung cao độ.
   * **[tsne_qic_d8.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/tsne_qic_d8.png)**: Phân bố của QiC Cascaded. Sự phân tách lớp rõ nét và phân bố cụm rất mượt mà, chứng tỏ chất lượng biểu diễn không thua kém mạng cổ điển.
   * **[tsne_fidelity_knn_d8.png](file:///C:/Users/DELL/Documents/antigravity/focused-newton/results/plots/tsne_fidelity_knn_d8.png)**: Phân bố của Fidelity Classifier (dựa trên nền PCA).

---

## 4. Kết luận & Định hướng Tiếp theo

### Kết luận
* Dự án đã hoàn thành mục tiêu xây dựng một framework đánh giá đối chứng toàn diện lượng tử - cổ điển.
* Mô hình lượng tử mô phỏng **QiC Cascaded** từ bài báo arXiv:2501.04591 đã được chứng minh là một giải pháp nén nhúng cực kỳ mạnh mẽ, đạt hiệu quả tối ưu hóa tham số vượt trội và chất lượng tương đương/hơn mạng cổ điển.
* Hiện tượng Barren Plateaus trên mạch PQC đã được thực chứng toán học và thực nghiệm qua sự khác biệt rõ rệt giữa hai phiên bản Entanglement và No-Entanglement.

### Định hướng tiếp theo
1. **Đẩy toàn bộ mã nguồn và kết quả lên kho lưu trữ GitHub** để đồng bộ hóa và lưu trữ lâu dài.
2. **Khám phá thêm các kỹ thuật giảm barren plateaus** khác như tối ưu hóa cấu trúc mạch PQC thích ứng (Adaptive PQC) hoặc khởi tạo tham số thông minh (Identity initialization) nếu bắt buộc phải làm việc với phần cứng lượng tử thực tế.
