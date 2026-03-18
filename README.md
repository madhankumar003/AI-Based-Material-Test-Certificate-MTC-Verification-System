# MTC Verification System

An intelligent document verification system that processes PDF and image documents, extracts text using OCR technology, and performs automated compliance checks with audit logging.

## 📋 System Architecture

The system follows a multi-stage pipeline architecture:

```
┌─────────────┐
│ PDF/Image   │
│   Input     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Gatekeeper  │──────────┐
└──────┬──────┘          │
       │                 │
       ├─────────────────┤
       │                 │
   Digital PDF      Scanned/Image
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│ pdfplumber  │   │ PaddleOCR   │
│             │   │PPStructureV3│
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                ▼
        ┌──────────────┐
        │ Compliance   │
        │    Check     │
        └───────┬──────┘
                │
     ┌──────────┼──────────┐
     │          │          │
     ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌─────────┐
│  PASS  │ │  FAIL  │ │ Manual  │
│ (Green)│ │  (Red) │ │ Review  │
└────┬───┘ └────┬───┘ └────┬────┘
     │          │           │
     └──────────┼───────────┘
                ▼
         ┌─────────────┐
         │  Audit Log  │
         └─────────────┘
```

### Pipeline Stages

1. **Gatekeeper**: Determines document type (digital PDF vs scanned/image)
2. **Text Extraction**:
   - Digital PDFs → `pdfplumber` for accurate text extraction
   - Scanned/Images → `PaddleOCR PPStructureV3` for OCR processing
3. **Compliance Check**: Validates extracted content against predefined standards
4. **Result Classification**:
   - ✅ **PASS**: Document meets all compliance requirements
   - ❌ **FAIL**: Document has compliance violations
   - ⚠️ **Manual Review**: Requires human verification
5. **Audit Log**: Records all processing results and decisions

## 🚀 Installation

### Prerequisites
- Anaconda or Miniconda installed
- Python 3.10 (recommended for PaddlePaddle compatibility)
- AMD Ryzen or CPU-based system (CPU version)

### Setup Instructions

**Step 1: Create Conda Environment**
```bash
conda create -n mtc_verify python=3.10 -y
```

**Step 2: Activate Environment**
```bash
conda activate mtc_verify
```

**Step 3: Install PaddlePaddle (CPU Version)**
```bash
python -m pip install paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

**Step 4: Install PaddleOCR with Document Parser**
```bash
pip install "paddleocr[doc-parser]"
```

**Step 5: Install Other Dependencies**
```bash
pip install streamlit pandas pillow pathlib pdfplumber beautifulsoup4
```

## 📦 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── gatekeeper.py          # Document type classifier
├── ocr_engine.py          # OCR processing engine
├── processor.py           # Text extraction processor
├── compliance_checker.py  # Compliance validation logic
├── standards.py           # Compliance standards definitions
└── README.md              # Project documentation
```

## 🎯 Features

- **Dual Processing Modes**: Handles both digital and scanned documents
- **Intelligent Classification**: Automatic document type detection
- **Advanced OCR**: PaddleOCR PPStructureV3 for complex layouts
- **Compliance Validation**: Automated standard checking
- **Audit Trail**: Complete logging of all operations
- **Web Interface**: User-friendly Streamlit dashboard

## 💻 Usage

1. **Activate the environment**:
   ```bash
   conda activate mtc_verify
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface** at `http://localhost:8501`

4. **Upload documents** and view compliance results

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime environment |
| PaddlePaddle | 3.2.0 | Deep learning framework |
| PaddleOCR | latest | OCR engine |
| Streamlit | latest | Web interface |
| pdfplumber | latest | PDF text extraction |
| Pandas | latest | Data processing |
| Pillow | latest | Image handling |
| BeautifulSoup4 | latest | HTML parsing |





## 📧 Contact

**Madhankumar S.** * **Email:** [madhankumarsaravanan3@gmail.com](mailto:madhankumarsaravanan3@gmail.com)
* **Phone:** +91 63832 91545
* **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/madhankumar-s-97727625a/)

---

**Note**: This system is optimized for CPU-based processing. For GPU acceleration, install the appropriate CUDA-enabled PaddlePaddle version.
