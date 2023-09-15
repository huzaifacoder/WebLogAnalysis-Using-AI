import streamlit as st

st.markdown('<p style="font-size:100px;">Contact Links</p>', unsafe_allow_html=True)

url_li = "https://www.linkedin.com/in/huzaifa-gulzar-shaikh-1a92991b8/"
url_git = "https://github.com/huzaifacoder/"
st.write('<p style="font-size:35px;"><a href="%s">LinkedIn</a></p>' % url_li, unsafe_allow_html=True)
st.write('<p style="font-size:35px;"><a href="%s">Github</a></p>' % url_git, unsafe_allow_html=True)
st.markdown('<p style="font-size:35px;">Email: huzaifashaikhpython@gmail.com</p>', unsafe_allow_html=True)