const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const title_area = document.getElementById('title-drop-area');
const resultDiv = document.getElementById('result');
const resultDiv1 = document.getElementById('result1');
const overlay = document.getElementById('overlay');

const efficientnetb4 = document.getElementById('efficientnetb4');
const mobilenetv3large = document.getElementById('mobilenetv3large');
let tempHeatmap_e = null;
let tempHeatmap_m = null;

// Kéo thả ảnh vào khu vực
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = '#f0f0f0';
});

dropArea.addEventListener('dragleave', () => {
    dropArea.style.backgroundColor = '';
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

// Xử lý khi chọn tệp ảnh
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Hàm xử lý ảnh tải lên
function handleFileUpload(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        preview.src = e.target.result;

        // Khi ảnh đã tải xong, điều chỉnh kích thước vùng chứa
        preview.onload = () => {
            dropArea.style.width = 'auto';
            dropArea.style.height = '500px';
            dropArea.style.border = 'none'; // Ẩn viền khung chứa
            preview.style.display = 'block';
        };
    };
    resultDiv.style.display = 'none';
    resultDiv1.style.display = 'none';
    title_area.style.display = 'none';

    reader.readAsDataURL(file);
    uploadImage();
}


// Kích hoạt chọn ảnh khi nhấp vào khu vực
dropArea.addEventListener('click', () => {
    fileInput.click();
});



function uploadImage() {
    const fileInput = document.getElementById('file-input');
    const overlay = document.getElementById('overlay');
    const preview = document.getElementById('preview');
    const file = fileInput.files[0];

    if (!file) {
        alert("Vui lòng chọn một ảnh trước!");
        return;
    }

    const validFormats = ['image/jpeg', 'image/png'];
    if (!validFormats.includes(file.type)) {
        alert("Vui lòng chọn tệp ảnh có định dạng JPG hoặc PNG!");
        return;
    }

    if (file.size > 5 * 1024 * 1024) { // 5MB
        alert("Tệp quá lớn. Vui lòng chọn tệp nhỏ hơn 5MB.");
        return;
    }

    // Hiển thị overlay và vô hiệu hóa nút khi đang xử lý
    overlay.style.display = 'flex';
    // btn_kiemtra.disabled = true;

    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .finally(() => {
        // Đảm bảo luôn ẩn overlay và bật nút "Kiểm tra" khi hoàn tất
        overlay.style.display = 'none';
        // btn_kiemtra.disabled = false;
    })
    .then(data => {

        mobilenetv3large.innerHTML = `${data.result}`;
        efficientnetb4.innerHTML = `${data.result1}`;
        // Cập nhật hoặc giữ nguyên hình ảnh
        if (data.heatmap) {
            // Cập nhật preview bằng heatmap
            preview.src = `data:image/png;base64,${data.heatmap}`;
            // tempHeatmap_e = `data:image/png;base64,${data.heatmap1}`;
            tempHeatmap_m = `data:image/png;base64,${data.heatmap}`;
        } 
        else {
            tempHeatmap_m = preview.src;
        }
        if (data.heatmap1) {
            // Cập nhật preview bằng heatmap
            preview.src = `data:image/png;base64,${data.heatmap}`;
            tempHeatmap_e = `data:image/png;base64,${data.heatmap1}`;
        } 
        else {
            tempHeatmap_e = preview.src;
        }

        preview.style.display = 'block'; // Đảm bảo ảnh luôn được hiển thị

        efficientnetb4.style.display = 'block';
        mobilenetv3large.style.display = 'block';

        syncDivWidth();
        const button = document.getElementById("mobilenetv3large");
        if (button) {
            highlightButton(button);
        } else {
            console.error("Không tìm thấy nút với ID 'mobilenetv3large'");
        }

    })
    .catch(error => {
        console.error('Error:', error);
        alert("Đã xảy ra lỗi trong quá trình xử lý!");
    });
}


document.getElementById('efficientnetb4').addEventListener('click', function () {
    preview.src = tempHeatmap_e;
});

document.getElementById('mobilenetv3large').addEventListener('click', function () {
    preview.src = tempHeatmap_m;
});



const container = document.querySelector(".container-change");

function checkAndShift() {
    if (mobilenetv3large.innerText.trim() !== "") { 
        // Nếu #result có nội dung, thêm class để dịch phải
        container.classList.add("shift-right");
    } else {
        // Nếu không có nội dung, trả lại vị trí ban đầu
        container.classList.remove("shift-right");
    }
}

// Gọi hàm kiểm tra mỗi khi nội dung #result thay đổi
const observer = new MutationObserver(checkAndShift);
observer.observe(mobilenetv3large, { childList: true, subtree: true });



function syncDivWidth() {
    const div1 = document.getElementById("efficientnetb4");
    const div2 = document.getElementById("mobilenetv3large");

    // Hiển thị tạm thời để tính toán kích thước
    div1.style.display = "inline-block";
    div2.style.display = "inline-block";

    // Tính chiều rộng lớn nhất
    const maxWidth = Math.max(div1.offsetWidth, div2.offsetWidth);

    // Gán chiều rộng lớn nhất cho cả hai div
    div1.style.width = maxWidth + 4 + "px";
    div2.style.width = maxWidth + 4 + "px";
}


// Thêm sự kiện click vào từng nút
document.getElementById("efficientnetb4").addEventListener("click", function() {
    highlightButton(this);
});

document.getElementById("mobilenetv3large").addEventListener("click", function() {
    highlightButton(this);
});

// Hàm thêm class 'active' vào nút được chọn
function highlightButton(button) {
    // Xóa class 'active' khỏi tất cả các nút
    const buttons = document.querySelectorAll("button"); // Chọn tất cả nút button
    buttons.forEach(btn => btn.classList.remove("active"));

    // Thêm class 'active' cho nút vừa được chọn
    button.classList.add("active");
}