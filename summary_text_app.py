import streamlit as st

from goose3 import Goose

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import T5Tokenizer
from transformers import T5Model, T5ForConditionalGeneration

# modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
tokenizer_pt_cstnews = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
model_pt_cstnews = T5ForConditionalGeneration.from_pretrained('phpaiola/ptt5-base-summ-xlsum')

st.title("Gerador de manchete a partir do texto da notícia 📰")

# Opção para a origem da notícia
opcao = st.selectbox("🎲 Escolha um opção:", ["Texto", "Link"], index=0, help="Opção de qual origem é a notícia, se é um texto de notícia ou um link para extrair o texto da notícia.")
if opcao == "Texto":
  form = st.form(key='my_form')
  texto_noticia = form.text_area(label="📝 Insira uma notícia: ", value="", height=300, placeholder='Insira o texto aqui...')
  submit_button = form.form_submit_button(label='Resumir 🎉')
if opcao == "Link":
  form = st.form(key='my_form')
  url_noticia = form.text_input(label="🌐 Insira um link: ", value="", placeholder='Informe o link aqui...')
  submit_button = form.form_submit_button(label='Resumir 🎉')

# Gerador da manchete após a escolha da origem
if submit_button and opcao == "Texto" and texto_noticia != "":
  with st.spinner('Resumindo...'):
    # modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
    inputs_pt_xlsum = tokenizer_pt_cstnews.encode(texto_noticia, max_length=512, truncation=True, return_tensors='pt')
    summary_ids_pt_xlsum = model_pt_cstnews.generate(inputs_pt_xlsum, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary_pt_cstnews = tokenizer_pt_cstnews.decode(summary_ids_pt_xlsum[0])

    st.write("#### Modelo 'phpaiola/ptt5-base-summ-cstnews' ✨")
    st.success(body=summary_pt_cstnews)
if submit_button and opcao == "Link" and url_noticia != "":
  with st.spinner('Extraindo notícia...'):
    g = Goose()
    article = g.extract(url=url_noticia)
    titulo_noticia_link = article.title
    texto_noticia_link = article.cleaned_text
    g.close()

    st.write('### Notícia 📄')
    st.info(body=texto_noticia_link)
    st.write('##### Título Original')
    st.info(body=titulo_noticia_link)

  with st.spinner('Resumindo...'):
    # modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
    inputs_pt_xlsum = tokenizer_pt_cstnews.encode(texto_noticia_link, max_length=512, truncation=True, return_tensors='pt')
    summary_ids_pt_xlsum = model_pt_cstnews.generate(inputs_pt_xlsum, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary_pt_cstnews = tokenizer_pt_cstnews.decode(summary_ids_pt_xlsum[0])

    st.write("#### Modelo 'phpaiola/ptt5-base-summ-cstnews' ✨")
    st.write('##### Título Gerado')
    st.success(body=summary_pt_cstnews)
else:
  st.warning(body="Insira uma notícia!!!", icon="⚠")


st.write("by Rafhael Martins")