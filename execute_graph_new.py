"""
New Execute Graph Function using Unified Agent
Replaces the complex state machine with the intelligent unified agent
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_agent import chat_with_unified_agent

def execute_graph_new(user_input: str, user_token: str = "user123") -> str:
    """
    New simplified execute_graph function using the unified agent
    This replaces the old complex state machine approach
    
    Args:
        user_input: User's message
        user_token: User authentication token
        
    Returns:
        AI response string (compatible with existing API)
    """
    
    # Get response from unified agent
    result = chat_with_unified_agent(user_input, user_token)
    
    # Extract the response message
    response = result.get("message", "Sorry, I couldn't process your request.")
    
    # Add redirect URL if there's a successful transaction
    if (result.get("response_type") == "transaction" and 
        result.get("status") == "success" and 
        result.get("redirect_url")):
        response += f" Navigate to: {result['redirect_url']}"
    
    return response

# Backward compatibility function
def execute_graph(thread_id: str, user_input: str, user_token: str = None) -> str:
    """
    Backward compatible version of execute_graph
    Maintains the same function signature as the old approach
    """
    # Use the new unified agent approach
    return execute_graph_new(user_input, user_token or "user123")

if __name__ == "__main__":
    # Test the new approach
    print("🧪 Testing New Unified Execute Graph")
    print("=" * 50)
    
    test_cases = [
        "hello",
        "What services does LB Finance offer?",
        "Send 500 to alice", 
        "to bob",
        "200",
        "I need help with banking"
    ]
    
    for test in test_cases:
        print(f"\nInput: '{test}'")
        response = execute_graph_new(test)
        print(f"Response: {response[:100]}...")
        print("-" * 30)
