import streamlit as st
import sys
import os

st.set_page_config(page_title="Memory Test", page_icon="🧪", layout="centered")

st.title("🧪 Render Memory Test")
st.write("यह ऐप बहुत हल्का है — कोई भारी लाइब्रेरी लोड नहीं करता।")
st.write("अगर यह पेज दिख रहा है, तो आपकी Render सर्विस ठीक चल रही है।")
st.write("समस्या भारी मॉडल (sentence-transformers / chromadb) के लोड होने में है।")

st.divider()

# सिस्टम की जानकारी दिखाएं
st.subheader("System Info")
st.write(f"Python version: `{sys.version.split()[0]}`")
st.write(f"Current directory: `{os.getcwd()}`")

# RAM की स्थिति (अगर psutil उपलब्ध है)
try:
    import psutil
    mem = psutil.virtual_memory()
    st.write(f"Total RAM: `{round(mem.total / 1024**3, 2)} GB`")
    st.write(f"Available RAM: `{round(mem.available / 1024**3, 2)} GB`")
    st.write(f"Used RAM: `{mem.percent}%`")
except Exception as e:
    st.info(f"psutil उपलब्ध नहीं: {e}")

st.divider()

# एक बटन जो सिर्फ टेक्स्ट दिखाए
if st.button("टेस्ट बटन दबाएं"):
    st.success("🎉 बटन काम कर रहा है! Streamlit सही से चल रहा है।")