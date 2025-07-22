import streamlit as st
import sys
import os
from unittest.mock import patch

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from unified_agent import chat_with_unified_agent
except ImportError as e:
    st.error(f"Error importing unified agent: {e}")
    st.stop()


# ─── Mock API Call Behaviors ─────────────────────────────────────────

def mock_get_user_primary_account(user_token):
    return "123456"  # Dummy source account ID

def mock_check_account_existence(user_token, account):
    if isinstance(account, dict):
        account = account.get("name", "UNKNOWN")
    
    # Simulate some accounts not existing
    invalid_accounts = ["badaccount", "invalid", "nonexistent", "fake"]
    if account.lower() in invalid_accounts:
        return None
    
    return account

def mock_check_account_balance(user_token, amount):
    # Simulate insufficient balance for large amounts
    if float(amount) > 1000:
        return "Insufficient balance"
    
    return str(amount)

def mock_process_transfer(user_token, source, destination, amount):
    transaction_id = f"TXN{hash(f'{source}{destination}{amount}') % 10000:04d}"
    return f"Transfer successful: Transaction ID {transaction_id}"


# ─── Streamlit UI Configuration ──────────────────────────────────────

st.set_page_config(
    page_title="LB Finance Unified Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    .system-info {
        background-color: #d1ecf1;
        border: 1px solid #b8daff;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session State ────────────────────────────────────────

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "streamlit-session"
if 'user_token' not in st.session_state:
    st.session_state.user_token = "user123"
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}


# ─── Main UI Layout ──────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 LB Finance Unified Agent</h1>
    <h3>Intelligent Banking Assistant with Natural Language Understanding</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with controls and information
with st.sidebar:
    st.header("🔧 Session Controls")
    
    # Agent Status
    st.success("✅ Using Unified Agent - Natural language understanding & intelligent conversation routing")
    
    st.divider()
    
    # Session Information
    st.info(f"""
    **Session Info:**
    - Session ID: `{st.session_state.thread_id}`
    - User Token: `{st.session_state.user_token}`
    - Messages: {len(st.session_state.chat_history)}
    """)
    
    # Reset conversation button
    if st.button("🔄 Reset Conversation", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.conversation_count = 0
        st.session_state.conversation_context = {}
        st.session_state.thread_id = f"streamlit-session-{len(st.session_state.chat_history)}"
        st.rerun()
    
    st.divider()
    
    # Testing scenarios
    st.header("🧪 Test Scenarios")
    st.markdown("""
    **Try these examples:**
    
    **General Chat:**
    - `hello` → Natural greeting
    - `What services does LB Finance offer?`
    - `Tell me about your branches`
    
    **Transaction Requests:**
    - `Send 500 to Alice` → Complete transfer
    - `to Bob` → Ask for amount
    - `300` → Ask for destination
    - `Transfer money to John`
    
    **Mixed Conversations:**
    - `What is LB Finance and can I send money?`
    - `I need help with transfer to Sarah`
    
    **Error Scenarios:**
    - `badaccount`, `invalid` → Account not found
    - Amount > $1000 → Insufficient balance
    """)
    
    st.divider()
    
    # Download conversation
    if st.session_state.chat_history:
        conversation_text = "\n".join([
            f"{'User' if msg['type'] == 'user' else 'Bot'}: {msg['content']}"
            for msg in st.session_state.chat_history
        ])
        st.download_button(
            "📥 Download Conversation",
            conversation_text,
            file_name=f"banking_chat_{st.session_state.thread_id}.txt",
            mime="text/plain"
        )

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Banking Chat")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="system-info">
                👋 Welcome to LB Finance Banking Chatbot!<br>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f"""
                <div style="text-align: right; margin: 1rem 0;">
                    <div class="user-message">
                        👤 You: {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Determine message style based on content
                if "success" in message['content'].lower() and "transferred" in message['content'].lower():
                    message_class = "success-message"
                    icon = "🎉"
                elif "missing" in message['content'].lower() or "provide" in message['content'].lower():
                    message_class = "system-info"
                    icon = "❓"
                elif "not in your" in message['content'].lower() or "insufficient" in message['content'].lower():
                    message_class = "error-message"
                    icon = "❌"
                else:
                    message_class = "bot-message"
                    icon = "🤖"
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div class="{message_class}">
                        {icon} Bot: {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    st.divider()
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([4, 1])
        
        with col_input:
            user_input = st.text_input(
                "Type your message:",
                placeholder="e.g., 'hello', 'Send 500 to Alice', or 'What services do you offer?'",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_button = st.form_submit_button("Send 📤", type="primary", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.chat_history.append({
            'type': 'user',
            'content': user_input.strip(),
            'timestamp': st.session_state.conversation_count
        })
        
        # Show processing indicator
        with st.spinner("🤖 Processing your request..."):
            try:
                # Use the unified agent approach
                with patch("unified_agent.UnifiedLBFinanceAgent._get_user_primary_account", return_value="123456"), \
                     patch("controllers.apiController.APIController.check_account_existence", side_effect=mock_check_account_existence), \
                     patch("controllers.apiController.APIController.check_account_balance", side_effect=mock_check_account_balance), \
                     patch("controllers.apiController.APIController.process_transfer", side_effect=mock_process_transfer):
                    
                    # Get response from unified agent with conversation context
                    result = chat_with_unified_agent(user_input.strip(), st.session_state.user_token, st.session_state.conversation_context)
                    response = result.get("message", "Sorry, I couldn't process your request.")
                    
                    # Update conversation context for next turn
                    st.session_state.conversation_context['last_response'] = result
                    st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                    
                    # Reset context after successful transactions to avoid carrying over completed transaction data
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and
                        not result.get("needs_more_info", False)):
                        st.session_state.conversation_context = {}
                    
                    # Add redirect URL if there's a successful transaction
                    if (result.get("response_type") == "transaction" and 
                        result.get("status") == "success" and 
                        result.get("redirect_url")):
                        response += f" Navigate to: {result['redirect_url']}"
                
                # Add bot response to chat
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': response,
                    'timestamp': st.session_state.conversation_count
                })
                
                st.session_state.conversation_count += 1
                
            except Exception as e:
                st.session_state.chat_history.append({
                    'type': 'bot',
                    'content': f"❌ Error: {str(e)}. Please try again with a different input.",
                    'timestamp': st.session_state.conversation_count
                })
        
        # Rerun to update the chat display
        st.rerun()

with col2:
    st.header("📊 Statistics")
    
    # Conversation statistics
    total_messages = len(st.session_state.chat_history)
    user_messages = len([msg for msg in st.session_state.chat_history if msg['type'] == 'user'])
    bot_messages = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot'])
    
    # Success/error counts
    success_count = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot' and 'success' in msg['content'].lower()])
    error_count = len([msg for msg in st.session_state.chat_history if msg['type'] == 'bot' and ('missing' in msg['content'].lower() or 'insufficient' in msg['content'].lower() or 'not in your' in msg['content'].lower())])
    
    st.metric("Total Messages", total_messages)
    st.metric("User Messages", user_messages)
    st.metric("Bot Messages", bot_messages)
    st.metric("Successful Transfers", success_count, delta=success_count if success_count > 0 else None)
    st.metric("Errors/Requests", error_count, delta=-error_count if error_count > 0 else None)
    
    # Quick action buttons
    st.header("⚡ Quick Actions")
    
    quick_actions = [
        ("� Say Hello", "hello"),
        ("ℹ️ LB Finance Info", "What services does LB Finance offer?"),
        ("�💰 Send $500", "Send 500 to Alice"),
        ("👤 To Bob", "to Bob"),
        ("🏦 Transfer Help", "I need help with transfer"),
        ("❌ Test Error", "badaccount"),
        ("💸 Large Amount", "1500 to John")
    ]
    
    for label, action in quick_actions:
        if st.button(label, key=f"quick_{action}", use_container_width=True):
            # Simulate user input
            st.session_state.chat_history.append({
                'type': 'user',
                'content': action,
                'timestamp': st.session_state.conversation_count
            })
            
            # Process the action
            with st.spinner("Processing..."):
                try:
                    # Use unified agent
                    with patch("unified_agent.UnifiedLBFinanceAgent._get_user_primary_account", return_value="123456"), \
                         patch("controllers.apiController.APIController.check_account_existence", side_effect=mock_check_account_existence), \
                         patch("controllers.apiController.APIController.check_account_balance", side_effect=mock_check_account_balance), \
                         patch("controllers.apiController.APIController.process_transfer", side_effect=mock_process_transfer):
                        
                        result = chat_with_unified_agent(action, st.session_state.user_token, st.session_state.conversation_context)
                        response = result.get("message", "Sorry, I couldn't process your request.")
                        
                        # Update conversation context
                        st.session_state.conversation_context['last_response'] = result
                        st.session_state.conversation_context['turn'] = st.session_state.conversation_count
                        
                        # Reset context after successful transactions
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and
                            not result.get("needs_more_info", False)):
                            st.session_state.conversation_context = {}
                        
                        if (result.get("response_type") == "transaction" and 
                            result.get("status") == "success" and 
                            result.get("redirect_url")):
                            response += f" Navigate to: {result['redirect_url']}"
                    
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': response,
                        'timestamp': st.session_state.conversation_count
                    })
                    
                    st.session_state.conversation_count += 1
                    
                except Exception as e:
                    st.session_state.chat_history.append({
                        'type': 'bot',
                        'content': f"❌ Error: {str(e)}",
                        'timestamp': st.session_state.conversation_count
                    })
            
            st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;">
    <strong>LB Finance Unified Agent</strong> — Intelligent Banking Assistant<br>
    Built with Streamlit | Powered by Unified Agent Platform<br>
    Leveraging Natural Language Understanding and Secure Transaction Processing<br><br>
    <em>This application is currently in experimental development and may be subject to changes.</em><br>
    &copy; 2025 LB Finance PLC. All rights reserved.
</div>
""", unsafe_allow_html=True)

