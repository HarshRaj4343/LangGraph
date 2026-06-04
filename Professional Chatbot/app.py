import streamlit as st

# Configure page
st.set_page_config(page_title="HarshGPT")

# 1. Initialize chat history at the very top
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Render Custom CSS + HTML Layout
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');

/* Top-left company name */
.company-name {
    position: fixed;
    top: 49px;
    left: 40px;
    font-size: 32px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}

/* Welcome banner */
.banner-name {
    position: fixed;
    top: 350px;
    left: 650px;
    font-size: 32px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# 3. Always display the company name logo
st.markdown('<div class="company-name">HarshGPT</div>', unsafe_allow_html=True)

# 4. CRITICAL: Redisplay past messages from chat history first
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. ONLY display the welcome banner if no messages exist in history
if not st.session_state.messages:
    st.markdown('<div class="banner-name">Ready to dive in, Harsh?</div>', unsafe_allow_html=True)

# 6. Get new user input
prompt = st.chat_input("Ask Anything.....")

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    response = f"You said: {prompt}"

    
    with st.chat_message("assistant"):
        st.markdown(response)
    # Save assistant response into chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()