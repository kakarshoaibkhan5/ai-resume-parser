# AI Resume Parser - 100% Open Source, Runs Locally

> Parse, summarise and score any resume in seconds - No data leaves your machine.

**Live Demo:** https://ai-resume-parser-kakarshoaib.streamlit.app

### Features
- 📄 **PDF to Structured JSON** - PyMuPDF text layer extraction
- 🧠 **BERT NER** - Extract name, email, phone, location, links
- 📝 **BART Summarization** - facebook/bart-large-cnn - 3-bullet executive summary
- 🎯 **Job Match Scoring** - 34 skills across 9 categories
- 🔒 **Privacy First** - 3 local models, runs on CPU, 100% offline capable

### Tech Stack
Python, Streamlit, PyMuPDF, Transformers (BERT, BART), PyTorch, Scikit-learn

### How it works
1. Drop PDF -> PyMuPDF extracts text
2. BERT NER identifies entities (131ms)
3. BART generates summary
4. Skill encoder matches job description

Built by Shoaib Kakar | v1.0.0 Open Source
