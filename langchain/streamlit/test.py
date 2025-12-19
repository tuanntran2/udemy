import streamlit as st
from agent import agent

st.title("LangChain Agent Test")
question = st.text_input("Enter your question for the agent:")
btn = st.button("Ask Agent")
if btn:
    session_id = "streamlit_session"
    result = agent.invoke({"input": question})
    st.write("Agent Response:")
    st.write(result)
