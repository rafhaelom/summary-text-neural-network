import streamlit as st

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import T5Tokenizer
from transformers import T5Model, T5ForConditionalGeneration

# modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
tokenizer_pt_cstnews = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
model_pt_cstnews = T5ForConditionalGeneration.from_pretrained('phpaiola/ptt5-base-summ-xlsum')

st.title("Gerador de manchete a partir do texto de notícia 📰")

form = st.form(key='my_form')
manchete_texto = form.text_area(label="📝 Insira uma notícia: ", value="", height=300, placeholder='Escreva aqui...')
submit_button = form.form_submit_button(label='Resumir 🎉')

if submit_button and manchete_texto != "":
  with st.spinner('Resumindo...'):

    # modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
    inputs_pt_xlsum = tokenizer_pt_cstnews.encode(manchete_texto, max_length=512, truncation=True, return_tensors='pt')
    summary_ids_pt_xlsum = model_pt_cstnews.generate(inputs_pt_xlsum, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary_pt_cstnews = tokenizer_pt_cstnews.decode(summary_ids_pt_xlsum[0])

    st.write("#### Modelo 'phpaiola/ptt5-base-summ-cstnews' ✨")
    st.info(body=summary_pt_cstnews)
else:
  st.warning(body="Insira uma notícia!!!", icon="⚠")


st.write("by Rafhael Martins")