# Nghiên cứu Chi tiết & Cơ sở Toán học: Nén Nhúng Quantum-Classical

Tài liệu này cung cấp mô tả chi tiết, chặt chẽ về mặt toán học và lý thuyết cho toàn bộ các phương pháp trong nghiên cứu **Nén biểu diễn nhúng lượng tử - cổ điển (Quantum-Classical Embedding Compression)**. Tài liệu được thiết kế nhằm phục vụ trực tiếp cho việc **viết bài báo khoa học (paper)**, giải thích đầy đủ từ các nền tảng cơ bản đến các định lý nâng cao.

---

## 1. Cơ sở Tiền xử lý & Trích xuất Đặc trưng (PhoBERT CLS)

### 1.1. Bản chất của Vector Nhúng [CLS]
Trong các kiến trúc dựa trên Transformer như BERT hay RoBERTa (ở đây là PhoBERT), câu đầu vào $\mathbf{S}$ gồm các từ tố (tokens) $[T_1, T_2, \dots, T_L]$ được đưa qua các lớp chú ý đa đầu (Multi-Head Attention). Token đặc biệt `[CLS]` (Classification) luôn được thêm vào đầu chuỗi:
$$\mathbf{S}_{\text{tokens}} = [\text{[CLS]}, T_1, T_2, \dots, T_L]$$

Qua nhiều tầng chú ý, biểu diễn của token `[CLS]` tại tầng cuối cùng, ký hiệu là $\mathbf{u} \in \mathbb{R}^{d}$ (với $d = 768$ đối với PhoBERT-base), đóng vai trò tích hợp thông tin ngữ nghĩa toàn cục của cả câu thông qua cơ chế self-attention:
$$\mathbf{u} = \text{Transformer}(\mathbf{S}_{\text{tokens}})_{0, :}$$

### 1.2. Phép phân tách từ tố tiếng Việt (Word Segmentation)
Tiếng Việt là ngôn ngữ đa âm tiết, ranh giới từ không trùng với khoảng trắng. Để PhoBERT hoạt động chính xác, câu phải được phân tách từ trước khi token hóa. Chúng ta sử dụng thư viện `underthesea` dựa trên mô hình Trường ngẫu nhiên điều kiện (Conditional Random Fields - CRF) hoặc mạng Bi-LSTM-CRF để tối ưu hóa ranh giới từ.
Một câu tiếng Việt $\mathbf{S}_{\text{raw}}$ được phân tách thành chuỗi các từ $\mathbf{S}_{\text{seg}}$:
$$\mathbf{S}_{\text{seg}} = \text{WordSegmentation}(\mathbf{S}_{\text{raw}}) = [w_1, w_2, \dots, w_M]$$
Ví dụ: `"học sinh học sinh học"` $\to$ `"học_sinh học môn sinh_học"`.

---

## 2. Các Phương pháp Nén Nhúng Cổ điển (Baselines)

### 2.1. Phân tích Thành phần Chính (PCA - Principal Component Analysis)
PCA thực hiện một phép biến đổi tuyến tính trực giao để chuyển đổi tập hợp các vector nhúng $\mathbf{u} \in \mathbb{R}^{d}$ thành không gian mới $\mathbf{z} \in \mathbb{R}^{d'}$ ($d' \ll d$) sao cho các thành phần mới tuần tự có phương sai lớn nhất.
1. Tính vector trung bình: $\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{u}^{(i)}$
2. Tính ma trận hiệp phương sai: $\mathbf{\Sigma} = \frac{1}{N}\sum_{i=1}^N (\mathbf{u}^{(i)} - \boldsymbol{\mu})(\mathbf{u}^{(i)} - \boldsymbol{\mu})^T$
3. Giải bài toán trị riêng và vector riêng: $\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$, sắp xếp $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$
4. Ma trận chiếu tuyến tính được tạo từ $d'$ vector riêng hàng đầu: $\mathbf{W}_{\text{PCA}} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_{d'}]^T \in \mathbb{R}^{d' \times d}$
5. Phép nén: $\mathbf{z} = \mathbf{W}_{\text{PCA}}(\mathbf{u} - \boldsymbol{\mu})$

### 2.2. Mạng Tự mã hóa Đan xen Phân loại (Joint Autoencoder)
Mạng Autoencoder phi tuyến tính học đồng thời không gian biểu diễn nén (latent space) $\mathbf{z} \in \mathbb{R}^{d'}$ thông qua hai hàm liên kết:
* **Encoder ($f_{\boldsymbol{\theta}_e}$):** $\mathbf{z} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2)$
* **Decoder ($g_{\boldsymbol{\theta}_d}$):** $\mathbf{\hat{u}} = \sigma(\mathbf{W}_4 \cdot \text{ReLU}(\mathbf{W}_3 \mathbf{z} + \mathbf{b}_3) + \mathbf{b}_4)$

Để đảm bảo $\mathbf{z}$ vừa giữ thông tin cấu trúc của $\mathbf{u}$ vừa tối ưu cho nhiệm vụ phân loại cảm xúc (3 lớp), chúng ta cực tiểu hóa hàm mất mát kết hợp (Joint Loss Function):
$$\mathcal{L}_{\text{AE}}(\boldsymbol{\theta}_e, \boldsymbol{\theta}_d, \boldsymbol{\theta}_c) = \mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{\hat{y}}) + \lambda_{\text{recon}} \mathcal{L}_{\text{MSE}}(\mathbf{u}, \mathbf{\hat{u}})$$

Trong đó:
1. **Reconstruction Loss (MSE):** $\mathcal{L}_{\text{MSE}} = \frac{1}{M}\sum_{k=1}^M \|\mathbf{u}^{(k)} - \mathbf{\hat{u}}^{(k)}\|^2_2$
2. **Classification Loss with Class Weights (Cross Entropy):**
   $$\mathcal{L}_{\text{CE}} = - \frac{1}{M}\sum_{k=1}^M \sum_{c=1}^3 w_c \cdot y_{c}^{(k)} \log(\hat{y}_{c}^{(k)})$$
   Trọng số động của lớp $c$ để giải quyết mất cân bằng dữ liệu: $w_c = \frac{M}{3 \cdot M_c}$ (với $M_c$ là số mẫu thuộc lớp $c$).
3. **Classifier Head ($h_{\boldsymbol{\theta}_c}$):** $\mathbf{\hat{y}} = \text{Softmax}(\mathbf{W}_c \mathbf{z} + \mathbf{b}_c)$

---

## 3. Mạch Lượng tử Biến phân (PQC) & Lý thuyết Barren Plateaus

### 3.1. Mô hình Mạch Lượng tử Biến phân lai (Hybrid Classical-Quantum PQC)
Mạch PQC biểu diễn dữ liệu nén dưới dạng trạng thái lượng tử của hệ $N$ qubit ($N = d'$). Trạng thái lượng tử của hệ được ký hiệu là $|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle \in \mathcal{H} \cong \mathbb{C}^{2^N}$, được chuẩn bị từ trạng thái gốc $|0\rangle^{\otimes N}$ qua hai toán tử đơn trị (unitaries):
$$|\psi(\mathbf{x}, \boldsymbol{\theta})\rangle = U(\boldsymbol{\theta}) W(\mathbf{x}) |0\rangle^{\otimes N}$$

#### 3.1.1. Toán tử mã hóa dữ liệu (Angle Embedding)
Toán tử $W(\mathbf{x})$ thực hiện xoay góc độc lập trên từng qubit trên trục $Y$ của quả cầu Bloch:
$$W(\mathbf{x}) = \bigotimes_{j=1}^N R_Y(x_j)$$
Với cổng xoay $R_Y(\phi)$ được định nghĩa toán học:
$$R_Y(\phi) = \exp\left( -i \frac{\phi}{2} Y \right) = \cos\left(\frac{\phi}{2}\right) I - i \sin\left(\frac{\phi}{2}\right) Y = \begin{pmatrix} \cos(\frac{\phi}{2}) & -\sin(\frac{\phi}{2}) \\ \sin(\frac{\phi}{2}) & \cos(\frac{\phi}{2}) \end{pmatrix}$$

#### 3.1.2. Mạch lượng tử biến phân đan xen (CNOT Ring Ansatz)
Ansatz $U(\boldsymbol{\theta})$ gồm các lớp xoay tham số xen kẽ các lớp đan xen (entanglement) để tạo mối liên kết lượng tử:
$$U(\boldsymbol{\theta}) = \prod_{l=1}^L \left( U_{\text{ent}} \cdot \bigotimes_{j=1}^N R_Y(\theta_{j, l}) \right)$$

Lớp đan xen toàn cục sử dụng chuỗi cổng CNOT nối vòng (Ring Entanglement):
$$U_{\text{ent}} = \text{CNOT}_{N, 1} \prod_{j=1}^{N-1} \text{CNOT}_{j, j+1}$$
Cổng CNOT tác động lên không gian Hilbert 2-qubit:
$$\text{CNOT} = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

#### 3.1.3. Phép đo (Measurement)
Đo giá trị kỳ vọng của toán tử Pauli-Z trên từng qubit thu được vector đặc trưng $\mathbf{z} \in [-1, 1]^N$:
$$z_j = \langle Z_j \rangle = \langle \psi(\mathbf{x}, \boldsymbol{\theta}) | Z_j | \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle$$
Trong đó toán tử Pauli-Z đơn qubit là:
$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### 3.2. Chứng minh Toán học về Hiện tượng Barren Plateaus
Hàm chi phí lượng tử có dạng $E(\boldsymbol{\theta}) = \langle \psi(\mathbf{x}, \boldsymbol{\theta})| H |\psi(\mathbf{x}, \boldsymbol{\theta})\rangle$, với $H$ là toán tử Hermitian biểu diễn Hamiltonian đo đạc.
Theo định lý **McClean và cộng sự (2018)**, nếu mạch lượng tử biến phân $U(\boldsymbol{\theta})$ đủ sâu để đạt được cấu trúc cấu thành một 2-design lượng tử (tiệm cận độ đo ngẫu nhiên Haar trên nhóm đơn trị $U(2^N)$):

1. **Trị kỳ vọng của gradient theo các tham số $\theta_k$ bằng 0:**
   $$\mathbb{E}_{\boldsymbol{\theta} \sim \text{Haar}} \left[ \frac{\partial E(\boldsymbol{\theta})}{\partial \theta_k} \right] = 0$$
2. **Phương sai của gradient bị triệt tiêu lũy thừa theo số lượng qubit $N$:**
   $$\text{Var}_{\boldsymbol{\theta} \sim \text{Haar}} \left[ \frac{\partial E(\boldsymbol{\theta})}{\partial \theta_k} \right] = \mathbb{E}_{\boldsymbol{\theta}} \left[ \left( \frac{\partial E(\boldsymbol{\theta})}{\partial \theta_k} \right)^2 \right] \approx \frac{\text{Tr}(H^2) - \frac{1}{2^N}(\text{Tr}(H))^2}{2 \cdot (2^{2N} - 1)} \propto \mathcal{O}\left( \frac{1}{2^N} \right)$$

#### Hệ quả thực nghiệm đối chứng (Ring CNOT vs No-Entanglement)
* **PQC (Ring CNOT):** Lớp đan xen lượng tử $U_{\text{ent}}$ lan truyền nhanh chóng sự tương quan và rối lượng tử trên toàn bộ không gian Hilbert $2^N$ chiều. Hàm mất mát nhanh chóng phẳng hóa và triệt tiêu phương sai gradient ($\text{Var} \propto 2^{-8}$ đối với $N=8$). Mô hình gặp hiện tượng nghẽn đạo hàm (Gradient Stagnation) từ rất sớm và chỉ đạt **$72.55\%$** Accuracy.
* **PQC No-Entanglement:** Khi loại bỏ hoàn toàn các cổng CNOT, không gian Hilbert của mạch bị phân rã hoàn toàn thành tích tensor của $N$ không gian con 2 chiều độc lập:
  $$\mathcal{H} = \bigotimes_{j=1}^N \mathcal{H}_j \cong \bigotimes_{j=1}^N \mathbb{C}^2$$
  Trạng thái toàn hệ thống là trạng thái tách biệt hoàn toàn (fully separable state). Việc tối ưu hóa góc xoay trên mỗi qubit độc lập không chịu tác động của độ đo ngẫu nhiên Haar trên toàn nhóm $U(2^N)$ mà chỉ là các nhóm $U(2)$ độc lập. Phương sai gradient lúc này không chịu sự sụt giảm hàm mũ theo số qubit $N$:
  $$\text{Var}_{\theta} \left[ \frac{\partial E_j}{\partial \theta_{j, k}} \right] \propto \mathcal{O}(1)$$
  Kết quả là mô hình PQC No-Entanglement tránh được lỗi nghẽn cục bộ nghiêm trọng của Barren Plateaus, giúp bộ tối ưu hóa Adam tiếp tục học và đẩy độ chính xác lên **$80.10\%$** (tăng **7.55%** so với phiên bản có đan xen).

---

## 4. Phương pháp Nén Lượng tử Xếp tầng (QiC Cascaded)

Phương pháp này được hiện thực hóa dựa trên bài báo **arXiv:2501.04591** bằng một cấu trúc lượng tử mô phỏng hoàn toàn cổ điển (Classical Simulation) trên PyTorch.

```
Embedding PhoBERT (768 chiều)
        │  [Chiếu tuyến tính]
        ▼
Vector đặc trưng u_start (d_start chiều)
        │  [Mã hóa Bloch]
        ▼
Qubits khởi đầu |psi_j> (d_start qubit dạng 2D)
        │
        ▼  [Tầng nén xếp tầng 1: d_start -> d_start / 2]
        ├─► Quay đơn qubit: U(alpha, beta, gamma)
        ├─► Tác động cổng CNOT cục bộ từng cặp
        ├─► Phép đo Born & Bỏ qubit điều khiển (Partial Trace)
        │
        ▼  [Tầng nén xếp tầng 2: d_start/2 -> d_start/4]
        │   ... l lặp lại tuần tự cho đến d' ...
        ▼
Trạng thái lượng tử nén (d' qubit)
        │  [Trích xuất xác suất p(0) = |a|^2]
        ▼
Vector nén z (d' chiều) ───► [Lớp Linear + Softmax] ───► Dự đoán phân loại
```

### 4.1. Mã hóa Bloch (Bloch sphere Encoding)
Một đặc trưng số thực $u_j \in \mathbb{R}$ được chuyển đổi thành trạng thái của một qubit đơn $|\psi_j\rangle = a_j|0\rangle + b_j|1\rangle$ thông qua ánh xạ:
$$\theta_j = \tanh(u_j)\frac{\pi}{2} + \frac{\pi}{2} \in [0, \pi], \quad \phi_j = \pi$$
$$|\psi_j\rangle = \cos\left(\frac{\theta_j}{2}\right)|0\rangle + e^{i\pi}\sin\left(\frac{\theta_j}{2}\right)|1\rangle = \cos\left(\frac{\theta_j}{2}\right)|0\rangle - \sin\left(\frac{\theta_j}{2}\right)|1\rangle$$
Ma trận cột biểu diễn trạng thái qubit:
$$|\psi_j\rangle \equiv \begin{pmatrix} \cos(\frac{\theta_j}{2}) \\ -\sin(\frac{\theta_j}{2}) \end{pmatrix} = \begin{pmatrix} a_j \\ b_j \end{pmatrix}$$

### 4.2. Mạch Nén Từng bước (One Cascaded Step: $d \to d/2$)
Xét một cặp qubit kề nhau $(q_i, q_j)$ với trạng thái tương ứng là $|\psi_i\rangle = \begin{pmatrix} a_i \\ b_i \end{pmatrix}$ và $|\psi_j\rangle = \begin{pmatrix} a_j \\ b_j \end{pmatrix}$.

#### Bước 4.2.1. Phép xoay đơn qubit huấn luyện (Parameterized Rotation)
Áp dụng toán tử xoay đơn trị $U(\alpha, \beta, \gamma)$ độc lập lên mỗi qubit:
$$|\psi'_i\rangle = U(\alpha_i, \beta_i, \gamma_i)|\psi_i\rangle$$
$$|\psi'_j\rangle = U(\alpha_j, \beta_j, \gamma_j)|\psi_j\rangle$$
Vì chúng ta chỉ đo biên độ xác suất thực, phép xoay hiệu dụng thu gọn về ma trận xoay thực $R_Y(\beta)$:
$$\begin{pmatrix} a'_i \\ b'_i \end{pmatrix} = \begin{pmatrix} \cos(\frac{\beta_i}{2})a_i - \sin(\frac{\beta_i}{2})b_i \\ \sin(\frac{\beta_i}{2})a_i + \cos(\frac{\beta_i}{2})b_i \end{pmatrix}$$

#### Bước 4.2.2. Đan xen cục bộ & Phép đo Born (CNOT & Partial Trace)
Hệ trạng thái chung của cặp qubit trước đan xen là trạng thái tích:
$$|\Psi\rangle = |\psi'_i\rangle \otimes |\psi'_j\rangle = a'_i a'_j |00\rangle + a'_i b'_j |01\rangle + b'_i a'_j |10\rangle + b'_i b'_j |11\rangle$$

Áp dụng cổng CNOT với qubit $i$ làm điều khiển (control) và qubit $j$ làm mục tiêu (target) ($|10\rangle \leftrightarrow |11\rangle$):
$$|\Psi_{CNOT}\rangle = a'_i a'_j |00\rangle + a'_i b'_j |01\rangle + b'_i b'_j |10\rangle + b'_i a'_j |11\rangle$$

Thực hiện phép đo Born trên qubit điều khiển $i$ và loại bỏ nó (tương đương phép lấy vết riêng phần - Partial Trace trên không gian của qubit $i$).
Mật độ xác suất để thu được qubit $j$ ở các trạng thái $|0\rangle$ và $|1\rangle$:
$$p(0) = \text{Tr}(|0\rangle\langle 0|_i \otimes I_j \cdot |\Psi_{CNOT}\rangle\langle \Psi_{CNOT}|) = |a'_i a'_j|^2 + |b'_i b'_j|^2$$
$$p(1) = \text{Tr}(|1\rangle\langle 1|_i \otimes I_j \cdot |\Psi_{CNOT}\rangle\langle \Psi_{CNOT}|) = |a'_i b'_j|^2 + |b'_i a'_j|^2$$

#### Bước 4.2.3. Tái mã hóa (Re-encoding)
Qubit $j$ sau khi đo được chuẩn hóa và re-encode thành qubit nén mới kế thừa thông tin của cả hai:
$$|\psi_{\text{new}}\rangle = \sqrt{p(0)}|0\rangle + \sqrt{p(1)}|1\rangle$$
Quá trình này giảm số lượng qubit đi một nửa sau mỗi tầng ($d \to d/2$). Sau $\log_2(d_{\text{start}}/d')$ tầng, hệ thống thu được $d'$ qubit. Vector nén cuối cùng $\mathbf{z} \in [0, 1]^{d'}$ được trích xuất bằng biên độ $|a_k|^2$ của các qubit còn lại:
$$z_k = p_k(0) = |a_{k, \text{final}}|^2$$

---

## 5. Phân loại dựa trên Độ tương hợp Lượng tử (Fidelity Classifier)

### 5.1. Định nghĩa toán học
Xét trạng thái nén của mẫu thử nghiệm $|\Phi\rangle$ và trạng thái nguyên mẫu (prototype) của lớp $c$, ký hiệu là $|\Psi_c\rangle$. Cả hai đều là trạng thái lượng tử tách biệt hoàn toàn (fully separable states):
$$|\Phi\rangle = \bigotimes_{j=1}^{d'} |\phi_j\rangle, \quad |\Psi_c\rangle = \bigotimes_{j=1}^{d'} |\psi_{c, j}\rangle$$

Độ tương đồng lượng tử (Uhlmann's Fidelity) giữa hai trạng thái thuần khiết này là:
$$F(|\Phi\rangle, |\Psi_c\rangle) = |\langle \Phi | \Psi_c \rangle|^2$$
Do tính chất tách biệt của trạng thái, phép tính toán $2^{d'}$ chiều phức tạp được phân rã thành tích của các phép tính 2 chiều tuyến tính cực kỳ nhanh chóng trên phần cứng cổ điển:
$$F(|\Phi\rangle, |\Psi_c\rangle) = \prod_{j=1}^{d'} |\langle \phi_j | \psi_{c, j} \rangle|^2$$

Thay tích vô hướng của hai qubit đơn lẻ $|\langle \phi_j | \psi_{c, j} \rangle| = \cos\left( \frac{\theta_{\phi, j} - \theta_{\psi, j}}{2} \right)$ vào công thức:
$$F(|\Phi\rangle, |\Psi_c\rangle) = \prod_{j=1}^{d'} \cos^2\left( \frac{\theta_{\phi, j} - \theta_{\psi, c, j}}{2} \right)$$

### 5.2. Thuật toán Phân loại
1. **Tính toán Nguyên mẫu cổ điển (Class Prototypes):**
   Với mỗi lớp $c \in \{1, 2, 3\}$, tính vector trung bình từ tập huấn luyện:
   $$\mathbf{m}_c = \frac{1}{N_c}\sum_{i \in \text{Class } c} \mathbf{z}^{(i)}$$
2. **Mã hóa Lượng tử nguyên mẫu:**
   Chuyển đổi $\mathbf{m}_c$ thành các góc Bloch $\boldsymbol{\theta}_c$ để tạo trạng thái nguyên mẫu $|\Psi_c\rangle$.
3. **Quy tắc Quyết định (Decision Rule):**
   Mẫu thử nghiệm $\mathbf{z}_{\text{test}}$ được ánh xạ thành góc $\boldsymbol{\theta}_{\text{test}}$ và phân loại vào lớp có Fidelity cao nhất:
   $$c^* = \arg\max_{c} F(|\Phi_{\text{test}}\rangle, |\Psi_c\rangle) = \arg\max_{c} \prod_{j=1}^{d'} \cos^2\left( \frac{\theta_{\text{test}, j} - \theta_{c, j}}{2} \right)$$
