from pyChatGPT import ChatGPT 
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests

url = 'https://analisis-metacognitivo2.aegcloud.pro/'

valor_ingresado = st.text_input("Ingrese un valor")  # Obtener el valor ingresado por el usuario

data = {'foo': valor_ingresado}  # Agregar el valor ingresado al diccionario 'data'

response = requests.post(url, data=data)

if response.status_code == 200:
    st.write('Solicitud POST exitosa')
    st.write('Contenido de la respuesta:')
    st.write(response.text)
else:
    st.write('Error en la solicitud POST:', response.status_code)
    
#LEER Y CLASIFICAR LAS RESPUESTAS
data = pd.read_csv(r'objeto_si.csv')
adfin_answers, asir_answers, daw_answers, mark_answers, patro_answers, vestu_answers = [], [], [], [], [], []

for index, row in data.iterrows():
    if row["Ciclo"] == "ADFIN":
        adfin_answers.append(str((row["objeto_si"]))) 
    elif row["Ciclo"] == "ASIR":
        asir_answers.append(str((row["objeto_si"]))) 
    elif row["Ciclo"] == "DAW":
        daw_answers.append(str((row["objeto_si"]))) 
    elif row["Ciclo"] == "MARK":
        mark_answers.append(str((row["objeto_si"]))) 
    elif row["Ciclo"] == "PATRO":
        patro_answers.append(str((row["objeto_si"]))) 
    elif row["Ciclo"] == "VESTU":
        vestu_answers.append(str((row["objeto_si"]))) 

person1_answers = []


for index, row in data.iterrows():
    if row["DNI"] == "72838728M":
        person1_answers.append(str((row["objeto_si"])))
a = """
    Un grado superior de desarrollo de aplicaciones web tiene como objetivo formar a profesionales capacitados para diseñar, desarrollar, implementar y mantener aplicaciones web, tanto en el ámbito empresarial como en el de los servicios y el comercio electrónico. Algunos de los objetivos específicos del grado superior en desarrollo de aplicaciones web pueden incluir:

Adquirir conocimientos técnicos en programación web y aplicaciones multimedia.
Desarrollar habilidades en la gestión de proyectos y el trabajo en equipo.
Conocer las principales tecnologías y herramientas utilizadas en el desarrollo web.
Adquirir conocimientos en bases de datos y sistemas de gestión de contenidos.
Desarrollar habilidades en el diseño y la arquitectura de aplicaciones web.
Adquirir conocimientos en seguridad informática y protección de datos personales.
Desarrollar habilidades en la resolución de problemas y la toma de decisiones.
En cuanto a las competencias que se esperan de los graduados en desarrollo de aplicaciones web, éstas pueden incluir:

Conocimientos técnicos en programación, diseño y arquitectura web.
Habilidad para trabajar en equipo y gestionar proyectos de desarrollo web.
Capacidad para adaptarse a nuevas tecnologías y herramientas de desarrollo web.
Habilidad para diseñar y desarrollar aplicaciones web eficientes y seguras.
Capacidad para analizar y resolver problemas en el desarrollo de aplicaciones web.
Habilidad para trabajar en entornos dinámicos y cambiantes.
Capacidad para comunicar y presentar de forma efectiva el trabajo desarrollado.
"""

#LLAMADA AL CHATGPT
# session_token = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..1K4xDB69QDCvq957.8XFvLu5dFg23jOdjkDyT-B_LE826oFzkcnRUJmx-poHDheX45HTf0m3cKKSgRp2B6QXxMR01ELGOHb0ZdeS5TGXC_8qyl9xTX1MvvFIkxLDVEc884xroPBFJdne2d-xoQrriAkDWZQFhE87tJSLlID-BZBKgUS_leaCbxJL87_KTxBKU4F_DNI-P_RMUL8ErLNZEFVs_CISJMMQLSpPA1GDAtecSPll55_FGuoNI3iYEYT-Rro3pFBOXdJhiEgmoKvWfVoItdN8NemVtXxXHFGl3XlZUgh5F7b6LT6id2MO5y5uZv_04lkw6mSl-Bh7ziBmXw2qtQY9vGX5s2p4SKI4CduaEMZtLslZlPM0p23fnGoIt2BYC7ijSw2nwqOLnl_axGJK0Sw1Jpmy5moNRs8yQcusQ2qMPl8g3r9WIosfuaoIz8qRiiP2nSzYUwfGI3-fRzsnXC3XtN-sfeywH3TFIMWo9MvKa0mU3SWfQZ8H2PZvXAUZ7-_j8Eopz6fYxpwAImJD1gIrG2JGTKyU4Mffh8_IBAo7yt9W8T6NLdXyCT9t5536i751Ga9CW6ahTBb7f3RWPwcsNnIB7VMwxwy996uwBquHiGWua-gepZw2PsO7yEQB3xZKCdafur-MegcxcWep_qbpmGo6-8AEGKLgLKD8Ed6MS4rnrcPLKfcHAvboO7SNmBuB4-lBUltaWTPDEfiXK25OXbwpQ7qychURy9OLd4fPuYtP3gwGOVi6k1Mni2rI4oa_XAhlJTvH6MMaYZxQHVXSTNOcgLXx9cz1JTqnkmk4mRHnvlj8uoyVszXQi2EFq1ozz4bxFCiir4wYzBdCwC2bp5S--i7E89xL3RQ8DvCMKO3q3Ro3nPU8hD4QItoCHgHaxpexrtiq_4feHmVl9A4cAFEkTGyjC4ZuNT0Ety0fsM0JtytFNiTnBHGqB7ZNOSLMyjNqEs7IpnBxRlzCB5afLDG5cP3ipOIMILSVyEv2je8yWSEx2E5ogSL-inO2p-EcThnT5KxySZMmCDT25qQmLI0Gk8afm8M--c1PUd3Z2ZXx50ouqvftZyEjlTocQfoAITVIUc6cCXhEuCsIL4RuyVvz5Ps73WuM0K9MmESn8iQRddz_03MxyHHsDdGoiT97TaGz-ivOjoO2eRdzOwU5k0JbJH5xZOWSvpVg2wnYWKLv2_gOjEMrTYxx_4kSBxpeYKyiIhFKug6nuRUmRwfCksPiWjNEVLjPo_x0_K_thH0jii76WQcq-224VibB3hAMTMxdr7aLVwqPJNVHOVI0Log4vJcledthlzilRGiw4kNBOYqo95DyYjXZJ5haKnUdsQrb9pwCBmXeK0PFxssZ904Wwpd22tH9w5ZvJZCz5p39tniakd9UeOHaPQmY3N26jzXfV4h1w3lkdzjrBEEMwxuUFjaaolQE6GKpswRiDdHZm4mbGOHkQYeMYebEVhy17r9drLvTc4QrlwNuP8HA4vfgQTUvqo64QM6RvIvqHftawkVazxNYUEhTmWsuUemZXI-GHLTbDrfD9BafGI9yk3hp8bG2u8R9ZvPZAA4R1wkBCQiY1BumfaGN49ETuQRJF0HTf4mRVJ9b6BSbgEO7tzAN2adQ0T22ePh2FkUqmOmDHHp_QwPFja5NCfLebcLyBLqxDkdcLANfpS0g-BkraV0ZpuU4_SZrb3Qu1Et8tnF5coVJwZWm6A-1PHhClHpf3KEz5F5MVfkAPCAiSMPqD8UaTCGWbMY7CHUav9IyuX7-uoAOzz9ZyoUQuDC9-OGHQb8x15XzsNWffCRpJwjWMBTRBB_rw5HwXuWBBWk-GWzXZtSHhWRXhouRhoUjKloqhgUfeNmL1gg4lusel5NQF6phwVek1V3oJknO1XezLjyVeio69_OOzkqosSkHs2ZskisqnFfL--LG0m5TO9-o88OYESeZIO4tQuUSN8HYxyqtaWo0iiJHxMDumo8fJypiR5z5L13aGrNA8ZPm2S_tg-Mmz34wqgLihFhgDRMMqu87dYXrF78oz3uKbYmhYsCk-jY519yUgwOfiC3CrfG6LqTcbWCVbmh-yogcstCV-nLfTeosGIZUHNI_H5didKNzh6hzUjoYHUhpiCDFD7mM89lm5EoeEa3S1ZNoAhO-4QXYIA5AajgxtES9SX7hPfP94zh9rm-l__9eJkWg7KOblWBYTq1eT71FxtN9fEIJZQ7pa9h1UEvgJL8aP2EJ3yWY5KFZ6GGcKpFf9x_omFxDxF8AdQ5n9uSKNcK0I8wz43FF8HRbWYWumUV3n7GpEZS7hGgB7ynRkYZ-X6ZtQj9VADJtUojMcAwYM7KdCqO1FBBKOd8McBirXkjzYfS3-LEjeYCdu67ogQYF3toxzR3Xhc_rcfcHnHh601I4opVb9mZBiws_d31146_CqJ-r5NF1PDnm9JfOURLua2ySMDd2B13uzP3L0BFveLR2Dq62BO2grKLEbn97__2HtESrVp-ozDqc9yLHSxhPjmWUAy9xCPDBlQNt8KrA.FxxuIKliXcdC8v7F1lX9sQ"
# api = ChatGPT(session_token)
# text = api.send_message(f'objetivos y competencias en un grado superior de desarrolllo de aplicaciones web')


#TOKENIZAR MENSAJE Y AÑADIR NUEVA FRASE
text = a.lower()
sent_tokens = nltk.sent_tokenize(text)
stopwords = stopwords.words('spanish')


#LIMPIAR FRASES
def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

all_similarities = []
for answer in daw_answers:
    sent_tokens.append(answer)
    cleaned = list(map(clean_string, sent_tokens))
    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()
    similarities = []
    for idx,vector in enumerate(vectors[:-1]):
        similarities.append(cosine_sim_vectors(vectors[-1], vectors[idx]))
    element = sent_tokens.pop()
    all_similarities.append(max(similarities))


#PROBAR CON RESPUESTAS DE PERSON1

person1_answers.append(text)
cleaned2 = list(map(clean_string, person1_answers))
vectorizer2 = CountVectorizer().fit_transform(cleaned2)
vectors2 = vectorizer2.toarray()
similarities2 = []
for idx,answer in enumerate(person1_answers[:-1]):
    similarities2.append(cosine_sim_vectors(vectors2[-1], vectors2[idx]))

st.header("Person 1")    
st.line_chart(data=similarities2)
