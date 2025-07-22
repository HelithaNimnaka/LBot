# 🏦 LB Finance Banking Chatbot - Streamlit Interface

A beautiful web interface for testing the human-in-the-loop banking transfer system.

## 🚀 Quick Start

### Option 1: Using the Batch File (Windows)
```bash
run_streamlit.bat
```

### Option 2: Using Command Line
```bash
streamlit run streamlit_banking_app.py
```

### Option 3: With Custom Port
```bash
streamlit run streamlit_banking_app.py --server.port 8502
```

## 🌐 Access the App

Once running, open your web browser and go to:
- **Local:** http://localhost:8501
- **Network:** http://your-ip:8501

## 🧪 Testing Scenarios

### 1. **Missing Account Name**
- Type: `500`
- Then: `Alice`
- **Result:** Bot remembers the amount and completes transfer

### 2. **Missing Transfer Amount**
- Type: `send to Bob`
- Then: `250`
- **Result:** Bot remembers the account and completes transfer

### 3. **Error Recovery - Invalid Account**
- Type: `300 to badaccount`
- Then: `John`
- **Result:** Bot detects invalid account, then accepts valid one

### 4. **Error Recovery - Insufficient Balance**
- Type: `1500 to Sarah`
- Then: `800`
- **Result:** Bot detects insufficient funds, then accepts lower amount

### 5. **Complete Transfer**
- Type: `300 to Mike`
- **Result:** Transfer completes immediately

## 🎯 Features

### 💬 **Interactive Chat Interface**
- Real-time conversation with the banking bot
- Human-in-the-loop interactions
- Context preservation across messages

### 📊 **Live Statistics**
- Message counts
- Success/error tracking
- Session information

### ⚡ **Quick Actions**
- Pre-defined test scenarios
- One-click testing buttons
- Common banking operations

### 🔧 **Session Management**
- Reset conversations
- Download chat history
- Session persistence

### 🎨 **Beautiful UI**
- Professional banking theme
- Mobile-responsive design
- Color-coded message types

## 🛠️ Technical Details

### **Mock Services**
- **Account Validation:** Simulates account existence checks
- **Balance Verification:** Simulates insufficient balance scenarios
- **Transfer Processing:** Simulates successful money transfers

### **Invalid Test Accounts**
These accounts will trigger "account not found" errors:
- `badaccount`
- `invalid`
- `nonexistent`
- `fake`

### **Balance Limits**
- Amounts ≤ $1000: ✅ Sufficient balance
- Amounts > $1000: ❌ Insufficient balance

## 📱 User Interface Overview

### **Main Chat Area**
- Send messages to the banking bot
- View conversation history
- Real-time response processing

### **Sidebar Controls**
- Session information
- Reset conversation
- Testing scenarios guide
- Download conversation history

### **Statistics Panel**
- Live message counts
- Success/error metrics
- Quick action buttons

## 🎉 Key Benefits

1. **🔄 Human-in-the-Loop:** Bot asks for missing information naturally
2. **🧠 Context Memory:** Remembers previous inputs across conversation turns
3. **🛡️ Error Recovery:** Gracefully handles validation errors
4. **💡 Smart Parsing:** Understands simple inputs like just "Alice" or "500"
5. **📱 User-Friendly:** Web interface instead of terminal commands

## 🚨 Troubleshooting

### **Port Already in Use**
```bash
streamlit run streamlit_banking_app.py --server.port 8502
```

### **Module Import Errors**
Make sure you're in the correct directory:
```bash
cd "d:\LB Finance\ChatBot\server - my  Copy"
```

### **Streamlit Not Found**
Install Streamlit:
```bash
pip install streamlit
```

## 📞 Support

For issues or questions about the banking chatbot system, check the logs in the Streamlit interface or terminal output.

---

**🏦 LB Finance Banking Chatbot** - Revolutionizing customer banking interactions with AI-powered human-in-the-loop conversations!
