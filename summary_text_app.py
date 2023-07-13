import streamlit as st

from goose3 import Goose

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from transformers import T5Tokenizer
from transformers import T5Model, T5ForConditionalGeneration

from rouge_score import rouge_scorer

# With `ttl`, objects in cache are removed after 24 hours.
@st.cache_resource(ttl=24*3600)
def get_models():
  '''
  modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-xlsum)
  '''
  tokenizer_pt_cstnews = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
  model_pt_cstnews = T5ForConditionalGeneration.from_pretrained('phpaiola/ptt5-base-summ-xlsum')
  return tokenizer_pt_cstnews, model_pt_cstnews

def limpa_frases_texto(texto):
  '''
  Função para remover frases/parágrafos contidos no texto.
  '''
  frase_1 = 'O formato de distribuição de notícias do Correio Braziliense pelo celular mudou. A partir de agora, as notícias chegarão diretamente pelo formato Comunidades, uma das inovações lançadas pelo WhatsApp. Não é preciso ser assinante para receber o serviço. Assim, o internauta pode ter, na palma da mão, matérias verificadas e com credibilidade. Para passar a receber as notícias do Correio, clique no link abaixo e entre na comunidade:'
  frase_2 = 'Apenas os administradores do grupo poderão mandar mensagens e saber quem são os integrantes da comunidade. Dessa forma, evitamos qualquer tipo de interação indevida. Caso tenha alguma dificuldade ao acessar o link, basta adicionar o número (61) 99666-2581 na sua lista de contatos.'
  frase_3 = 'Quer ficar por dentro sobre as principais notícias do Brasil e do mundo? Siga o Correio Braziliense nas redes sociais. Estamos no Twitter, no Facebook, no Instagram, no TikTok e no YouTube. Acompanhe!'
  frase_4 = 'As informações são do jornal O Estado de S. Paulo.'
  frase_5 = '• Blogs Redirect Novas temporadas de Outer banks, As five e Você se destacam no streaming em fevereiro'
  frase_6 = 'Receba direto no celular as notícias mais recentes publicadas pelo Correio Braziliense. É de graça. Clique aqui e participe da comunidade do Correio, uma das inovações lançadas pelo WhatsApp.'
  frase_7 = 'O Correio tem um espaço na edição impressa para publicar a opinião dos leitores. As mensagens devem ter, no máximo, 10 linhas e incluir nome, endereço e telefone para o e-mail sredat.df@dabr.com.br.'


  texto = texto.split("¶")
  texto = [frase.strip() for frase in texto if frase not in '']
  texto = [frase.replace(frase_1, '') for frase in texto]
  texto = [frase.replace(frase_2, '') for frase in texto]
  texto = [frase.replace(frase_3, '') for frase in texto]
  texto = [frase.replace(frase_4, '') for frase in texto]
  texto = [frase.replace(frase_5, '') for frase in texto]
  texto = [frase.replace(frase_6, '') for frase in texto]
  texto = [frase.replace(frase_7, '') for frase in texto]
  texto = [frase for frase in texto if frase not in '']
  texto = ' '.join(texto)
  texto = texto.split('•')
  texto = [frase.strip() for frase in texto if frase not in '']
  return ' '.join(texto)

tokenizer_pt_cstnews, model_pt_cstnews = get_models()

st.title("Gerador de manchete a partir do texto da notícia 📰")

# Opção para a origem da notícia
opcao = st.selectbox("🎲 Escolha um opção:", ["Texto", "Link"], index=1, help="Opção de qual origem é a notícia, se é um texto de notícia ou um link para extrair o texto da notícia.")
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
    # modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-xlsum)
    inputs_pt_xlsum = tokenizer_pt_cstnews.encode(texto_noticia, max_length=512, truncation=True, return_tensors='pt')
    summary_ids_pt_xlsum = model_pt_cstnews.generate(inputs_pt_xlsum, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary_pt_cstnews = tokenizer_pt_cstnews.decode(summary_ids_pt_xlsum[0]).replace('<pad> ', '').replace('</s>','')

    st.write("#### Modelo 'phpaiola/ptt5-base-summ-xlsum' ✨")
    st.success(body=summary_pt_cstnews)

if submit_button and opcao == "Link" and url_noticia != "":
  with st.spinner('Extraindo notícia...'):
    g = Goose()
    article = g.extract(url=url_noticia)
    titulo_noticia_link = article.title
    texto_noticia_link = article.cleaned_text
    g.close()

    texto_noticia_link = limpa_frases_texto(texto_noticia_link)

    st.write('### Notícia 📄')
    with st.expander("Veja a notícia"):
      st.info(body=texto_noticia_link)
    
    st.write('##### Título Original')
    st.info(body=titulo_noticia_link)

  with st.spinner('Resumindo...'):
    # modelo phpaiola/ptt5-base-summ-cstnews (https://huggingface.co/phpaiola/ptt5-base-summ-cstnews)
    inputs_pt_xlsum = tokenizer_pt_cstnews.encode(texto_noticia_link, max_length=512, truncation=True, return_tensors='pt')
    summary_ids_pt_xlsum = model_pt_cstnews.generate(inputs_pt_xlsum, max_length=256, min_length=32, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
    summary_pt_cstnews = tokenizer_pt_cstnews.decode(summary_ids_pt_xlsum[0]).replace('<pad> ', '').replace('</s>','')

    st.write("#### Modelo 'phpaiola/ptt5-base-summ-xlsum' ✨")
    st.write('##### Título Gerado')
    st.success(body=summary_pt_cstnews)

    # Avaliação do modelo métrica Rouge.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(titulo_noticia_link, summary_pt_cstnews)

    st.write('##### Resultado')
    st.success(body=scores)
else:
  st.warning(body="Insira uma notícia!!!", icon="⚠")


st.write("by Rafhael Martins")