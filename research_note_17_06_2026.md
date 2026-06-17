# Nhật ký Nghiên cứu Chi tiết: Nén Nhúng Quantum-Classical (17/06/2026)

Tài liệu này cung cấp một bản phân tích lý thuyết và toán học toàn diện, chi tiết từ gốc đến ngọn về dự án **Nén nhúng kết hợp Lượng tử - Cổ điển (Quantum-Classical Embedding Compression)**. Mục tiêu là giúp người đọc dù **chưa có kiến thức nền tảng về cơ học lượng tử hay học máy lượng tử (QML)** vẫn có thể hiểu rõ:
1. Bản chất toán học của các phương pháp.
2. Tại sao chúng ta lại thiết kế như vậy.
3. Chi tiết nguyên nhân thất bại của Parameterized Quantum Circuit (PQC) do hiện tượng nghẽn gradient (Barren Plateaus).
4. Sự khác biệt cốt lõi so với bài báo gốc [arXiv:2501.04591](https://arxiv.org/abs/2501.04591).

---

## 1. Giới thiệu Bài toán & Khái niệm Cơ bản

### 1.1. Bài toán Nén Nhúng (Embedding Compression)
Trong xử lý ngôn ngữ tự nhiên (NLP), mô hình **PhoBERT** biến đổi các câu văn tiếng Việt thành các vector số thực có số chiều rất lớn ($d = 768$). Mỗi vector này gọi là một **nhúng câu (sentence embedding)** $u \in \mathbb{R}^{768}$.
* **Khó khăn:** Lưu trữ và tính toán trên vector 768 chiều cực kỳ tốn bộ nhớ và tài nguyên khi triển khai thực tế.
* **Mục tiêu:** Nén vector nhúng này xuống không gian chỉ còn $d' = 8$ chiều nhưng phải bảo toàn tối đa thông tin để phân loại cảm xúc (Tích cực, Tiêu cực, Trung lập).

### 1.2. Biểu diễn Lượng tử cơ bản cho người mới bắt đầu
Trong máy tính lượng tử, đơn vị thông tin cơ bản là **Qubit** (thay vì Bit cổ điển chỉ nhận giá trị $0$ hoặc $1$).
Một trạng thái của một qubit đơn lẻ, ký hiệu là $|\psi\rangle$ (đọc là "Ket psi"), là một tổ hợp tuyến tính (chồng chập - superposition) của hai trạng thái cơ bản $|0\rangle$ và $|1\rangle$:
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$
Trong đó $\alpha, \beta \in \mathbb{C}$ là các số phức thỏa mãn điều kiện chuẩn hóa xác suất:
$$|\alpha|^2 + |\beta|^2 = 1$$
* $|\alpha|^2$ là xác suất đo được qubit ở trạng thái $|0\rangle$.
* $|\beta|^2$ là xác suất đo được qubit ở trạng thái $|1\rangle$.

### 1.3. Biểu diễn hình học: Quả cầu Bloch (Bloch Sphere)
Chúng ta có thể biểu diễn trạng thái của một qubit dưới dạng tọa độ cực trên một quả cầu đơn vị 3 chiều (Quả cầu Bloch) bằng công thức:
$$|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle$$
* $\theta \in [0, \pi]$ là góc cực (độ vĩ).
* $\phi \in [0, 2\pi]$ là góc phương vị (độ kinh).
* $i$ là đơn vị ảo ($i^2 = -1$).

---

## 2. Chi tiết Toán học của 3 Phương pháp Thử nghiệm

Chúng ta thực hiện nén vector nhúng $u = (u_1, u_2, \dots, u_{768})$ xuống vector nén $z = (z_1, z_2, \dots, z_8)$.

### 2.1. Phương pháp 1: Mạng tự mã hóa cổ điển (Autoencoder)
Mạng Autoencoder gồm hai phần huấn luyện đồng thời:
1. **Encoder ($E$):** Ánh xạ phi tuyến tính từ $768$ chiều về $8$ chiều:
   $$z = \sigma(W_e \cdot u + b_e)$$
2. **Decoder ($D$):** Tái tạo lại không gian gốc $768$ chiều từ $8$ chiều:
   $$\hat{u} = \sigma(W_d \cdot z + b_d)$$

Hàm mất mát tổng hợp (Joint Loss) được tối ưu hóa bằng thuật toán lan truyền ngược:
$$\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \lambda \mathcal{L}_{\text{classification}}$$
$$\mathcal{L} = \frac{1}{M}\sum_{k=1}^{M} \|u^{(k)} - \hat{u}^{(k)}\|^2 - \frac{1}{M}\sum_{k=1}^{M}\sum_{c=1}^{3} y_{c}^{(k)} \log(\hat{y}_{c}^{(k)})$$
* **Lý do thành công ($90.71\%$):** Không gian 8 chiều ở giữa bị ép phải giữ thông tin cấu trúc hình học của 768 chiều (để giải mã) đồng thời phải phân biệt được nhãn cảm xúc (để phân loại).

### 2.2. Phương pháp 2: Mạch lượng tử biến phân (PQC) của chúng ta
Để thực hiện phân loại lượng tử, chúng ta thiết kế một pipeline gồm 3 bước:

```
Nhúng cổ điển (8 chiều) ---> [ Mã hóa góc (Angle Encoding) ]
                                    │
                                    ▼
                             [ Trạng thái lượng tử ban đầu ]
                                    │
                                    ▼
                             [ Các lớp biến phân & Đan xen (CNOT) ]
                                    │
                                    ▼
                             [ Đo trị kỳ vọng (Pauli-Z) ] ---> Xác suất phân loại
```

#### Bước 2.2.1: Mã hóa góc (Angle Encoding)
Ánh xạ 8 đặc trưng cổ điển $z = (z_1, z_2, \dots, z_8)$ thành trạng thái lượng tử của 8 qubit thông qua các cổng xoay $R_Y$:
$$|z\rangle = \bigotimes_{j=1}^{8} R_Y(z_j)|0\rangle$$
Trong đó, cổng xoay $R_Y(\theta)$ được định nghĩa dưới dạng ma trận toán học:
$$R_Y(\theta) = \begin{pmatrix} \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\ \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) \end{pmatrix}$$

#### Bước 2.2.2: Lớp đan xen lượng tử (Entanglement Layers)
Để các qubit có thể trao đổi thông tin với nhau, chúng ta sử dụng cổng **CNOT (Controlled-NOT)** dạng vòng tròn (Ring entanglement). Cổng CNOT tác động lên 2 qubit (Qubit điều khiển $C$ và Qubit mục tiêu $T$):
* Nếu $C = |0\rangle$, qubit $T$ giữ nguyên.
* Nếu $C = |1\rangle$, qubit $T$ bị đảo ngược ($|0\rangle \leftrightarrow |1\rangle$).
Ma trận toán học của cổng CNOT:
$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

#### Bước 2.2.3: Đo giá trị kỳ vọng (Expectation Values)
Sau khi đi qua mạch biến phân với các góc xoay $\theta$ cần huấn luyện, chúng ta tiến hành đo toán tử Pauli-Z trên các qubit để thu được các giá trị thực trong khoảng $[-1, 1]$:
$$\langle Z_j \rangle = \langle \psi(z, \theta) | Z_j | \psi(z, \theta) \rangle$$
Các giá trị này sau đó được đưa qua một lớp tuyến tính cổ điển và hàm Softmax để tính xác suất phân loại lớp cảm xúc.

---

## 3. Bản chất Toán học của Hiện tượng Barren Plateaus (Tại sao PQC thất bại?)

Mô hình PQC của chúng ta bị kẹt ở độ chính xác **$44.50\%$** (chỉ nhỉnh hơn đoán ngẫu nhiên một chút) do hiện tượng **Barren Plateaus (Mặt phẳng cằn cỗi)**.

### 3.1. Định nghĩa toán học
Giả sử hàm mất mát của chúng ta là $E(\theta)$ phụ thuộc vào các tham số $\theta$ của mạch lượng tử. Trong quá trình huấn luyện, chúng ta cần tính đạo hàm $\frac{\partial E}{\partial \theta_k}$ để cập nhật trọng số.
Tuy nhiên, đối với một mạch lượng tử có tính đan xen cao (như mạch CNOT Ring của chúng ta) trên hệ thống $N$ qubit:
1. **Giá trị trung bình của đạo hàm bằng 0:**
   $$\mathbb{E}_{\theta}\left[ \frac{\partial E(\theta)}{\partial \theta_k} \right] = 0$$
2. **Phương sai của đạo hàm bị triệt tiêu theo hàm mũ của số lượng Qubit ($N$):**
   $$\text{Var}_{\theta}\left[ \frac{\partial E(\theta)}{\partial \theta_k} \right] \approx \mathcal{O}\left( \frac{1}{2^N} \right)$$

### 3.2. Hệ quả trực quan
Khi số lượng qubit $N = 8$, phương sai của gradient giảm đi $2^8 = 256$ lần. Nếu $N = 20$, phương sai giảm đi hơn $1.04 \times 10^6$ lần!
Bề mặt của hàm mất mát lúc này không còn các thung lũng hay đồi núi để đi xuống, mà trở thành một mặt phẳng hoàn toàn phẳng lặng:
* Đạo hàm tại mọi điểm đều cực kỳ nhỏ (vídụ: $\approx 10^{-7}$).
* Thuật toán tối ưu hóa không thể biết phải đi theo hướng nào để giảm lỗi. Kết quả là mô hình hoàn toàn ngừng học (Gradient Stagnation).

---

## 4. Phân tích Giải pháp của Bài báo arXiv:2501.04591

Bài báo **arXiv:2501.04591** đã giải quyết bài toán nén nhúng bằng một tư duy hoàn toàn khác, giúp tránh được hiện tượng Barren Plateaus và tăng tốc độ xử lý:

### 4.1. Trạng thái lượng tử tách biệt hoàn toàn (Fully Separable States)
Để nén và so sánh nhúng, bài báo không dùng mạch lượng tử biến phân sâu để phân loại trực tiếp. Họ ánh xạ vector nhúng $u = (u_1, \dots, u_n)$ thành một trạng thái lượng tử của $n$ qubit độc lập, **không đan xen (no entanglement)**:
$$|u\rangle = \bigotimes_{j=1}^{n} |u_j\rangle$$
Với mỗi qubit đơn lẻ $|u_j\rangle$ được biểu diễn trên quả cầu Bloch với tọa độ cực:
$$\theta_j = \tanh(u_j)\frac{\pi}{2} + \frac{\pi}{2}, \quad \phi_j = \pi$$
$$|u_j\rangle = \cos\left(\frac{\theta_j}{2}\right)|0\rangle + e^{i\pi}\sin\left(\frac{\theta_j}{2}\right)|1\rangle = \cos\left(\frac{\theta_j}{2}\right)|0\rangle - \sin\left(\frac{\theta_j}{2}\right)|1\rangle$$
* **Tại sao dùng $\tanh(\cdot)$?** Hàm Tang hyperbolic giới hạn giá trị của embedding cổ điển về khoảng $[-1, 1]$, từ đó đưa $\theta_j$ về đúng phạm vi vật lý $[0, \pi]$ của quả cầu Bloch.

### 4.2. Độ đo tương đồng dựa trên Uhlmann's Fidelity
Để so sánh hai câu có trạng thái lượng tử $|u\rangle$ và $|v\rangle$, thay vì dùng Cosine Similarity cổ điển, bài báo sử dụng **Độ tương hợp lượng tử (Fidelity)**:
$$F(|u\rangle, |v\rangle) = |\langle u | v \rangle|^2$$
Vì hai trạng thái lượng tử $|u\rangle$ và $|v\rangle$ là tách biệt hoàn toàn (không có rối lượng tử), chúng ta có thể tính toán Fidelity cực kỳ nhanh trên máy tính cổ điển bằng tích của từng cặp qubit đơn lẻ mà không cần dựng ma trận $2^n \times 2^n$:
$$F(|u\rangle, |v\rangle) = \prod_{j=1}^{n} |\langle u_j | v_j \rangle|^2$$
Trong đó tích vô hướng của hai qubit đơn lẻ được tính bằng công thức lượng giác đơn giản:
$$\langle u_j | v_j \rangle = \cos\left(\frac{\theta_{u,j}}{2}\right)\cos\left(\frac{\theta_{v,j}}{2}\right) + \sin\left(\frac{\theta_{u,j}}{2}\right)\sin\left(\frac{\theta_{v,j}}{2}\right) = \cos\left( \frac{\theta_{u,j} - \theta_{v,j}}{2} \right)$$
Do đó:
$$F(|u\rangle, |v\rangle) = \prod_{j=1}^{n} \cos^2\left( \frac{\theta_{u,j} - \theta_{v,j}}{2} \right)$$
* **Ưu điểm lớn:**
  1. Không có Barren Plateaus vì không có các lớp đan xen lượng tử (no entanglement layers).
  2. Độ phức tạp tính toán chỉ là tuyến tính $\mathcal{O}(n)$ thay vì hàm mũ $\mathcal{O}(2^n)$, rất thân thiện với phần cứng GPU cổ điển.

### 4.3. Cơ chế nén lượng tử từng bước (Cascaded Compression)
Để nén từ $d$ chiều xuống $d'$ chiều, bài báo đề xuất cấu trúc nén từng bước ghép cặp:
1. Gom 2 qubit cạnh nhau thành 1 cặp.
2. Áp dụng cổng kiểm soát hai qubit đơn giản để chuyển thông tin từ qubit thứ nhất sang qubit thứ hai.
3. Thực hiện phép chiếu/đo (Measurement) qubit thứ nhất và bỏ nó đi. Qubit thứ hai giữ lại thông tin nén của cả hai.
4. Lặp lại quá trình này tuần tự cho đến khi đạt số chiều mong muốn.

---

## 5. Bảng so sánh Kiến trúc lượng tử

| Tiêu chí | PQC (Thực nghiệm của chúng ta) | Thiết kế đề xuất bởi bài báo (arXiv:2501.04591) |
| :--- | :--- | :--- |
| **Đan xen lượng tử (Entanglement)** | Rất cao (CNOT Ring đan xen toàn bộ 8 qubit) | Không có (Fully Separable States) hoặc chỉ có cục bộ từng đôi một |
| **Độ dốc Gradient** | Bị triệt tiêu hoàn toàn (Barren Plateaus) | Được bảo toàn tốt, hội tụ cực kỳ nhanh |
| **Tốc độ mô phỏng cổ điển** | Rất chậm (phải tính toán ma trận trạng thái đầy đủ) | Rất nhanh (phân rã thành tích của các phép tính 2 chiều độc lập) |
| **Độ đo tương đồng** | Trị kỳ vọng qua lớp phân loại cổ điển | Uhlmann's Fidelity lượng tử trực tiếp giữa các trạng thái |
