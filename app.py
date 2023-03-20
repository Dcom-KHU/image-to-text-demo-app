from transformers import pipeline
import streamlit as st
from PIL import Image

captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")


st.title('Image To Text')

with st.form('app'):
    upload_file = st.file_uploader("이미지 파일을 올려주세요.")
    submit = st.form_submit_button('분석!')
    if submit:
        image = Image.open(upload_file)
        result = captioner(image)[0]['generated_text']
        st.image(image)
        st.subheader('분석 결과')
        st.text(result)


