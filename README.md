# 🌐 AI Regulation Chatbot

## 📜 Overview
The **AI Regulation Chatbot** is an innovative tool designed to help users understand the **European Commission's Artificial Intelligence regulations**. This chatbot provides an interactive way to access and explore important information regarding AI laws.

## 🌟 Features
- **Interactive Q&A**: Ask questions about AI regulations and get immediate responses.
- **Clear Explanations**: Receive straightforward answers based on the regulatory document.
- **User-Friendly Interface**: Built with **Streamlit** for easy navigation and a smooth user experience.

## 🚀 Getting Started

### ⚙️ Prerequisites
- **Python**: Version 3.8 or higher
- **Docker**: Optional, for containerized deployment

### 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/AI-Regulation-Chatbot.git
   cd AI-Regulation-Chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

3. **Create a `.env` file**: 
   Add your keys to the `.env` file in the root directory. The format should be:
   ```plaintext
   ACCESS_KEY="your_access_key_here"
   SECRET_KEY="your_secret_key_here"
   ```

### 💻 Running the Chatbot
To start the chatbot using Streamlit, run:
```bash
streamlit run main_script.py
```

### 🐳 Docker Deployment
To run the chatbot with Docker:
1. **Build the Docker image**:
   ```bash
   docker build -t ai-regulation-chatbot .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8501:8501 ai-regulation-chatbot
   ```

## 🛠️ Technologies Used
- **FastEmbed**: For efficient text embeddings.
- **ChromaDB**: As the vector store for searching information.
- **LangChain**: For powerful natural language processing capabilities.
- **Streamlit**: For building an intuitive user interface.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request. Make sure to follow the guidelines for contributions.

## 📄 License
This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments
- A special thanks to the **European Commission** for providing the AI Act document.
