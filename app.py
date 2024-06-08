# --------------------LIBRERÍAS----------------------------#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sns
import os
from plotly.subplots import make_subplots
from streamlit.components.v1 import declare_component
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

st.set_page_config(
    page_title="Titanic",
    layout="wide",
    initial_sidebar_state="expanded", #collapsed
    page_icon="🚢", 
    )

# Con el decorador cache_data solo cargamos los datos la primera vez que se carga la página
@st.cache_data (show_spinner="Cargando Datos...")  # 👈 Add the caching decorator
def load_data(url):
        df = pd.read_csv(url)
        return df
@st.cache_data (show_spinner="Cargando Datos...") 
def merge_data(df,dfPuerto):
        dfPuerto = dfPuerto.add_prefix('Embarked_')
        df = pd.merge(df, dfPuerto, left_on='Embarked', right_on='Embarked_Embarked', how='left')
        #borramos la columna de union del segundo dataframe
        df.drop(columns=['Embarked_Embarked','Embarked'],inplace=True)
        df= df.set_index('PassengerId')
        return df
@st.cache_data (show_spinner="Generando WordCloud...") 
def wordCloud(survived,colormap):      
    datos = df[df['Survived'] == survived]['Apellido']
    wordcloud = WordCloud(width=800, height=400,background_color=background_color, mask=mascara, colormap=colormap, random_state=random_state).generate(' '.join(datos))
    wordcloud_imagen = wordcloud.to_array()
    imagen = Image.fromarray(wordcloud_imagen)
    imagen.save(f'img/wordcloudSurived{survived}.png')
    return wordcloud_imagen
df = load_data("datos/titanic_procesado.csv")
dfPuerto = load_data("datos/titanic_puertos.csv")
df = merge_data(df,dfPuerto)

mascara = np.array(Image.open("img/ship.jpg"))
random_state = 42
background_color = 'white'
wordcloud_fallecidos =  wordCloud(survived=0,colormap='inferno')
wordcloud_supervivientes =  wordCloud(survived=1,colormap='viridis')


# --------------------SIDEBAR----------------------------#
st.sidebar.image("img/titanic.jpeg",use_column_width=True)
st.sidebar.markdown("""
---
""")

filtered_df = df

sexosUnicos = df["Sex"].unique()
puertosUnicos = df["Embarked_Name"].unique()
clasesUnicos = sorted(df["Pclass"].unique())
selected_sexo = st.sidebar.multiselect('Seleccionar género', sexosUnicos, sexosUnicos)
selected_puerto = st.sidebar.multiselect('Seleccionar puerto', puertosUnicos, puertosUnicos)
selected_clase = st.sidebar.multiselect('Seleccionar Clase', clasesUnicos, clasesUnicos)
# on = st.sidebar.toggle('Activar filtros para Gráficos')

# if on:
#     st.write('Feature activated!')

#* Filtrar el DataFrame con los datos seleccionados
filtered_df = filtered_df[filtered_df['Sex'].isin(selected_sexo)]
filtered_df = filtered_df[filtered_df['Embarked_Name'].isin(selected_puerto)]
filtered_df = filtered_df[filtered_df['Pclass'].isin(selected_clase)]
# st.sidebar.markdown("""
# ---
# """)

# --------------------TITLE----------------------------#
st.title("Análisis Titanic")
st.write(filtered_df)
# --------------------SECCION DE PESTAÑAS----------------------------#
tabGen, tabPuertos, tabSexo, tabEdades, tabNubes= st.tabs(
    [
        "1.Análisis General",
        "2.Análisis por Puerto",
        "3.Análisis por Clase y  Sexo",
        "4.Análisis por Edades",
        "5.Nubes de Palabras",
    ]
)
with tabGen:
    #*Generamos los datos con los que vamos a trabajar
    #* Cambiar los índices de los valores de la columna 'Survived' a 'Superviviente' y 'Fallecido'
    survived_counts = df['Survived'].map({1: 'Supervivientes', 0: 'Fallecidos'}).value_counts()
    pasajeros_totales = survived_counts.sum()

    #* Contar los valores de la columna 'Sex'
    sex_counts = df['Sex'].value_counts()

    #* Contar los valores de la columna 'Pclass'
    pclass_counts = df['Pclass'].value_counts()
    pclass_porcentajes = [round((pasajeros / pasajeros_totales) * 100,2) for pasajeros in pclass_counts]
    #* Contar los valores de la columna 'Embarked'
    embarked_counts = df['Embarked_Name'].value_counts()
    embarked_porcentajes = [round((pasajeros / pasajeros_totales) * 100,2) for pasajeros in embarked_counts]

    #* Contar los valores de la columna 'TipoTarifa'
    tpFare_counts = df['TpFare'].value_counts()

    #* Contar los valores de la columna 'Titulo'
    titulo_counts = df['Titulo'].value_counts()

    #* Contar los valores de de los pasajeros que tienen familia 
    pasajeros_con_familia = df.query('Familiares>0').shape[0]

    #print(f'{pasajeros_con_familia_vivos=}{pasajeros_con_familia_muertos=}')
    #*Generamos el gráfico con Subplots
    def generarGrafico():
            # Crear la figura con subtramas 
            fig = make_subplots(rows=3, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}],
                                                [{'type':'xy', 'colspan':2},None, {"rowspan": 2}],
                                                [{'colspan': 2},None , None]],
                                #subplot_titles=['Sobrevivientes', 'Género', 'Clase', 'Embarque'],
                                horizontal_spacing=0.06, vertical_spacing=0.06)

            # Agregar el gráfico Supervivencia
            fig.add_trace(go.Pie(labels=survived_counts.index, 
                                values=survived_counts.values, name='', showlegend=False), 1, 1)

            # Agregar el gráfico de pastel para los pasajeros que tienene familia 
            fig.add_trace(go.Pie(labels=['Sin Familia', 'Con Familia'], 
                                values=[pasajeros_totales - pasajeros_con_familia, pasajeros_con_familia],
                                name='', showlegend=False), 1, 2)

            # Agregar el gráfico de pastel para el género
            fig.add_trace(go.Pie(labels=sex_counts.index, 
                                values=sex_counts.values, name='', showlegend=False), 1, 3)


            # Agregar el gráfico de pastel para la clase
            fig.add_trace(go.Bar(x=embarked_counts.values , y=embarked_counts.index,
                                name='',showlegend=False,
                                text=[f'{valor}%' for valor in embarked_porcentajes],
                                marker_color=embarked_porcentajes,  # Utiliza los porcentajes como valores de color
                                marker_colorscale='viridis',
                                orientation='h'),2, 1)

            fig.add_trace(go.Bar(x=pclass_counts.index , y=pclass_counts.values,
                                name='',showlegend=False,
                                text=[f'{valor}%' for valor in pclass_porcentajes],
                                marker_color=pclass_porcentajes,  # Utiliza los porcentajes como valores de color
                                marker_colorscale='thermal'),2, 3)

            fig.add_trace(go.Scatter(x=df['Age'] , y=df['Titulo'], mode='markers', name='',showlegend=False,
                                marker=dict(color='blue')),3, 1)

            # Actualizar las etiquetas de cada gráfico de pastel
            fig.update_traces(textinfo='label+percent', row=1, col=1)
            fig.update_traces(textinfo='label+percent', row=1, col=2)
            fig.update_traces(textinfo='label+percent', row=1, col=3)
            #fig.update_traces(textinfo='label+percent', row=2, col=1)
            fig.update_traces(textinfo='label+percent', row=2, col=2)
            #fig.update_traces(text='percent', row=3, col=1)


            # Actualizar diseño de la figura para hacer los gráficos más grandes y reducir la separación
            fig.update_layout( height=800, width=1100,title_text=f'Análisis general del Titanic sobre pasajeros totales: {pasajeros_totales} ')


            # Actualizar nombres de ejes para el gráfico de dispersión
            fig.update_xaxes(title_text='Edad', row=3, col=1)
            return fig
    #*Mostramos el gráfico con Subplots
    st.plotly_chart(generarGrafico(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        #* Generamos el Donuts Tasa Supervivencia con y sin Familia
        def generarGrafico():
            sobrevivientes_con_familia = df.query('Survived == 1 & Familiares>0').shape[0]
            sobrevivientes_sin_familia = df[(df['Familiares'] == 0) & (df['Survived'] == 1)].shape[0]

            # Calcular los porcentajes de supervivencia
            total_sobrevivientes = sobrevivientes_con_familia + sobrevivientes_sin_familia
            porcentaje_con_familia = (sobrevivientes_con_familia / total_sobrevivientes) * 100
            porcentaje_sin_familia = (sobrevivientes_sin_familia / total_sobrevivientes) * 100

            # Crear el gráfico de donut
            fig = go.Figure(data=[go.Pie(labels=[f'Con Familia ({sobrevivientes_con_familia})', 
                                        f'Sin Familia ({sobrevivientes_sin_familia})'], 
                                values=[sobrevivientes_con_familia, sobrevivientes_sin_familia], hole=.3)])
            fig.update_layout(title='Tasa de Supervivencia: Pasajeros con y sin Familia',
                            legend=dict(orientation="h", yanchor="top", y=1.13, xanchor="left", x=.2))
            return fig
            #* Mostramos el Donuts Tasa Supervivencia con y sin Familia    
        st.plotly_chart(generarGrafico(), use_container_width=True)
    with col2:
        #* Generamos el Scatter Tarifa vs Clase
        def generarGrafico():
            fig = px.scatter(df.sort_values(by='Pclass'), x='TpFare', y='Pclass', title='Tarifa vs Clase',
                    labels={'TpFare': 'Tarifa', 'Pclass': 'Clase'},
                    size='Fare',  # Tamaño de las burbujas basado en la tarifa
                    hover_name='Pclass',  # Texto que aparece al pasar el ratón sobre las burbujas
                    color='Pclass',  # Color de las burbujas basado en la clase
                    opacity=0.5
                    )  # Opacidad de las burbujas
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.19, xanchor="left", x=.2))
            return fig   
        #* Mostramos el Scatter Tarifa vs Clase
        st.plotly_chart(generarGrafico(), use_container_width=True)      

    #* Generamos el mapa correlación
    def generarGrafico():
        columnas = ['Age', 'Fare', 'Survived', 'Pclass', 'Sex' , 'Familiares']

        # Crea un DataFrame con solo las columnas seleccionadas
        df_subset = df.loc[:,columnas]
        df_subset['Pclass'] = df['Pclass'].map({'1ª Clase':1, '2ª Clase':2, '3ª Clase':3})
        df_subset['Sex'] = df['Sex'].map({'Hombres':1, 'Mujeres':2})
        # Calcula la matriz de correlación
        correlation_matrix = df_subset.corr()

        # Crea el mapa de correlación
        fig = px.imshow(correlation_matrix,
                        labels=dict( color="Correlación"),
                        x=correlation_matrix.index,
                        y=correlation_matrix.columns,
                        width=800, height=600,
                        color_continuous_scale='spectral',
                        zmin=-1, zmax=1 ,text_auto=True,aspect="auto" )

        fig.update_layout(title='Mapa de Correlación',
                        xaxis_title='Columnas',
                        yaxis_title='Columnas')
        return fig
    #* Mostrar el mapa de correlación
    st.plotly_chart(generarGrafico())
with tabPuertos:
    #* Generamos los datos con los que vamos a trabajar
    viajeros = df.groupby(['Embarked_Name','Embarked_Latitud','Embarked_Longitud', 'Survived']).size().unstack(fill_value=0).reset_index()
    puertos = viajeros['Embarked_Name'].values
    puertos_Latitud = viajeros['Embarked_Latitud'].values
    puertos_Longitud = viajeros['Embarked_Longitud'].values
    viajeros_muertos = viajeros[0].values
    viajeros_vivos = viajeros[1].values
    viajeros_totales = viajeros_muertos + viajeros_vivos
    viajeros_general = (viajeros_muertos.sum()) + (viajeros_vivos.sum())
    viajeros_general_muertos = viajeros_muertos.sum()
    viajeros_general_vivos = (viajeros_vivos.sum())

    #* Generamos el Expander en el que vamos a desarrollar la explicación
    with st.expander("Explicación",expanded=True):
        st.write(f'''<p>El total de pasajeros embarcados fue de {viajeros_general}
                (Fallecidos {((viajeros_general_muertos/viajeros_general)*100):.2f}% | 
                Supervivientes {((viajeros_general_vivos/viajeros_general)*100):.2f}%)</p>''', unsafe_allow_html=True)
        columnas = st.columns(len(puertos))
        for indice, puerto in enumerate(puertos):
            with columnas[indice] :         
                st.write(f'''<u><b>{puerto}</b></u>''', unsafe_allow_html=True)
                st.write(f'''<p>El {(viajeros_muertos[indice]/viajeros_totales[indice]*100):.2f}% de los pasajeros embarcados fallecieron <br>
                        El {(viajeros_vivos[indice]/viajeros_totales[indice]*100):.2f}% de los pasajeros embarcados sobrevivieron <br></p>''', unsafe_allow_html=True)
                st.write(f'''<ul>
                            <li>Embarcaron: {viajeros_totales[indice]} ({((viajeros_totales[indice]/viajeros_general)*100):.2f}% total)</li>
                            <li>Supervivientes: {viajeros_vivos[indice]} ({((viajeros_vivos[indice]/viajeros_general_vivos)*100):.2f}% total)</li>
                            <li>Fallecidos: {viajeros_muertos[indice]} ({((viajeros_muertos[indice]/viajeros_general_muertos)*100):.2f}% total)</li>
                        </ul>''', unsafe_allow_html=True)
    #* Generamos el gráfico de Mapa
    def generarMapa() :
        #Generamos el Mapa
        bubble_scale = 80 

        #* Normalizar el tamaño de las burbujas
        valor_maximo = viajeros_totales.max()
        normalized_totales = viajeros_totales / valor_maximo  
        normalized_muertos = viajeros_muertos / valor_maximo  
        normalized_vivos = viajeros_vivos / valor_maximo  

        #* Crear el primer conjunto de datos para mostrar personas embarcadas
        scatter_map_embarcados = go.Scattermapbox(
            lat=puertos_Latitud,
            lon=puertos_Longitud,
            text=['<b>Embarcados:</b> ' + str(viajeros) + '<br><b>Puerto:</b> ' + puerto for puerto, viajeros in zip(puertos, viajeros_totales)],
            marker=dict(
                size=normalized_totales * bubble_scale,
                color='blue',  # Color para mostrar personas embarcadas
                opacity=0.7  # Opacidad de los marcadores
            ),
            #mode='markers',
            name='Embarcados'  # Nombre para la leyenda
        )

        #* Crear el segundo conjunto de datos para mostrar personas fallecidas
        scatter_map_fallecidos = go.Scattermapbox(
            lat=puertos_Latitud,
            lon=puertos_Longitud,
            
            text=['<b>Fallecidos:</b> ' + str(viajeros) + '<br><b>Puerto:</b> ' + puerto for puerto, viajeros in zip(puertos, viajeros_muertos)],
            marker=dict(
                size=normalized_muertos * bubble_scale,
                color='red',  # Color para mostrar personas fallecidas
                opacity=0.7  # Opacidad de los marcadores
            ),
            #mode='markers',
            name='Fallecidos'  # Nombre para la leyenda
        )

        #* Crear el tercer conjunto de datos para mostrar personas supervivientes
        scatter_map_supervivientes = go.Scattermapbox(
            lat=puertos_Latitud,
            lon=puertos_Longitud,
            text=['<b>Supervivientes:</b> ' + str(viajeros) + '<br><b>Puerto:</b> ' + puerto for puerto, viajeros in zip(puertos, viajeros_vivos)],
            marker=dict(
                size=normalized_vivos * bubble_scale,
                color='green',  # Color para mostrar personas fallecidas
                opacity=0.7  # Opacidad de los marcadores
            ),
            #mode='markers',
            name='Supervivientes'  # Nombre para la leyenda
        )

        #* Configurar el diseño del mapa
        layout = go.Layout(
            title='Embarcados, Fallecidos y Supervivientes por Puerto',
            mapbox=dict(
                center=dict(lat=(puertos_Latitud.mean() + puertos_Latitud.mean()) / 2),  # Calcular el centro del mapa
                style='open-street-map',  # Cambia el estilo del mapa según tus preferencias
                zoom=5
            ),
            height=600,  # Altura del gráfico en píxeles,
            legend=dict(x=0.6, y=1.18, orientation='h')  # Posiciona la leyenda en el centro superior
        )

        #* Crear la figura con los conjuntos de datos y el diseño
        fig = go.Figure(data=[scatter_map_embarcados, scatter_map_fallecidos,scatter_map_supervivientes], layout=layout)
        return fig
    #* Mostramos el gráfico de Mapa
    st.plotly_chart(generarMapa(), use_container_width=True)
    #* Generamos el Sunburst Pasajeros por Puerto de Embarque, Clase y Supervivencia
    def generarSunburst():
        # Construir el gráfico de sunburst
        # Agrupar por puerto de embarque, clase de pasajero y supervivencia y crea una nueva Pasajeros con la cantidad de ocurrencias
        viajeros = df.groupby(['Embarked_Name', 'Pclass', 'Survived','Sex',]).size().unstack(fill_value=0).stack().reset_index(name='Pasajeros')

        # Convertir la columna Survived a formato de texto
        viajeros['Survived'] = viajeros['Survived'].map({0: 'Fallecidos', 1: 'Supervivientes'})

        # Convertir la columna Pclass a formato de texto
        viajeros['Pclass'] = viajeros['Pclass']
        # Convertir la columna Pclass a formato de texto
        viajeros['Sex'] = viajeros['Sex']

        fig = px.sunburst(viajeros, path=['Embarked_Name', 'Pclass', 'Survived','Sex'], values='Pasajeros',
                        title='Pasajeros por Puerto de Embarque, Clase y Supervivencia',
                        height=700, color_continuous_scale='Pasajeros')
        return fig
    #* Mostramos el Sunburst Pasajeros por Puerto de Embarque, Clase y Supervivencia
    st.plotly_chart(generarSunburst(), use_container_width=True)
with tabSexo:
    col1, col2 = st.columns(2)
    with col1:
        #* Asignamos los datos con los que vamos a trabajar
        viajeros = (df.groupby(['Sex','Pclass']).size().unstack(fill_value=0).stack().reset_index(name='Pasajeros')).sort_values(by='Pasajeros', ascending=False)
        sexos =  viajeros['Sex'].values
        clases =  viajeros['Pclass'].values
        pasajeros =  viajeros['Pasajeros'].values

        with st.expander("Explicación Clases",expanded=True):
                st.write(f'''<p>El total de pasajeros embarcados fue de {viajeros_general}<br>
                        (Fallecidos {viajeros_general_muertos} | Supervivientes {viajeros_general_vivos})</p>''', unsafe_allow_html=True)

                filaClases = st.columns(len(clasesUnicos))
                for indice, clase in enumerate(clasesUnicos):
                    viajerosClase = viajeros.query(f'Pclass == "{clase}"')['Pasajeros'].sum()
                    viajerosClaseSupervivientes = df.query(f'Pclass == "{clase}" & Survived==1')['Survived'].count()
                    viajerosClaseFallecidos = df.query(f'Pclass == "{clase}" & Survived==0')['Survived'].count()
                    with  filaClases[indice] :         
                        st.write(f'''<u><b>{clase}</b></u>''', unsafe_allow_html=True)
                        st.write(f'''<ul>
                            <li>Embarcaron: {viajerosClase}</br> ({((viajerosClase/viajeros_general)*100):.2f}% total)</li>
                            <li>Supervivientes: {viajerosClaseSupervivientes}</br>({((viajerosClaseSupervivientes/viajeros_general_vivos)*100):.2f}% total)</li>
                            <li>Fallecidos: {viajerosClaseFallecidos}</br> ({((viajerosClaseFallecidos/viajeros_general_muertos)*100):.2f}% total)</li>
                        </ul>''', unsafe_allow_html=True)

        def generarGrafico():
            fig = go.Figure()

            for sexo in viajeros['Sex'].unique():
                clases2 = viajeros.loc[viajeros['Sex'] ==sexo]['Pclass']
                valores2 = viajeros.loc[viajeros['Sex'] ==sexo]['Pasajeros']
                fig.add_trace(go.Funnel(
                    name = sexo,
                    y = clases2,
                    x = valores2,
                    textinfo = "label+text+value+percent total"))  # ['label', 'text', 'percent initial', 'percent previous', 'percent total', 'value']

            fig.update_layout(title="Sexos por Clases",legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.2))
            return fig
        st.plotly_chart(generarGrafico(), use_container_width=True) 
    with col2:
        #* Asignamos los datos con los que vamos a trabajar
        viajeros = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0).reset_index()
        sexo =  viajeros['Sex'].values
        viajeros_muertos = viajeros[0].values
        viajeros_vivos = viajeros[1].values
        viajeros_totales = viajeros_muertos + viajeros_vivos

        with st.expander("Explicación Sexo",expanded=True):
                st.write(f'''<p>El total de pasajeros embarcados fue de {viajeros_general}<br>
                        (Fallecidos {viajeros_muertos.sum()} | Supervivientes {viajeros_vivos.sum()})</p>''', unsafe_allow_html=True)
                filaSexo = st.columns(len(sexosUnicos))
                for indice, sexo in enumerate(sexosUnicos):
                    viajerosSexo = df.query(f'Sex == "{sexo}"')['Survived'].count()
                    viajerosSexoSupervivientes = df.query(f'Sex == "{sexo}" & Survived==1')['Survived'].count()
                    viajerosSexoFallecidos = df.query(f'Sex == "{sexo}" & Survived==0')['Survived'].count()
                    with  filaSexo[indice] :         
                        st.write(f'''<u><b>{sexo}</b></u>''', unsafe_allow_html=True)
                        st.write(f'''<ul>
                            <li>Embarcaron: {viajerosSexo}</br> ({((viajerosSexo/viajeros_general)*100):.2f}% total)</li>
                            <li>Supervivientes: {viajerosSexoSupervivientes}</br>({((viajerosSexoSupervivientes/viajeros_general_vivos)*100):.2f}% total)</li>
                            <li>Fallecidos: {viajerosSexoFallecidos}</br> ({((viajerosSexoFallecidos/viajeros_general_muertos)*100):.2f}% total)</li>
                        </ul>''', unsafe_allow_html=True)
        def generarGrafico():
            fig = go.Figure()
            sexo =  viajeros['Sex'].values
            # Agregar trazos al gráfico
            fig.add_trace(go.Scatter(x=sexo, y=viajeros_totales, 
                            mode='lines+markers',
                            marker_color='blue',
                            name='Pasajeros'))

            fig.add_trace(go.Scatter(x=sexo, y=viajeros_vivos,      
                            mode='lines+markers',
                            marker_color='lightgreen',
                            name='Supervivientes'))

            fig.add_trace(go.Scatter(x=sexo, y=viajeros_muertos, 
                            mode='lines+markers',
                            marker_color='lightcoral',
                            name='Fallecidos'))
            
            fig.update_layout(title="Datos por Sexo",legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            return fig
        # * Mostrar el gráfico
        st.plotly_chart(generarGrafico(), use_container_width=True) 

    #* Agrupar por 'Pclass', 'Sex' y 'Survived' y contar la cantidad de pasajeros
    #* Generamos datos con los que vamos a trabajar
    viajeros = df.groupby(['Pclass', 'Sex', 'Survived']).size().unstack(fill_value=0)
    
    #* Renombrar las columnas para mayor claridad
    viajeros.columns = ['Fallecidos', 'Supervivientes']

    #* Resetear el índice para convertirlo en columnas
    viajeros = viajeros.reset_index()

    #* Definir un esquema de color personalizado
    colors = {'Fallecidos': 'rgba(255, 0, 0, 0.6)', 'Supervivientes': 'rgba(0, 255, 0, 0.6)'}

    def generarGrafico():
        fig = px.area(viajeros, x='Sex', y=['Fallecidos', 'Supervivientes'], color_discrete_map=colors,
                        title='Cantidad de fallecidos y supervivientes por clase y sexo',
                        labels={'value': 'Cantidad', 'Sex': 'Sexo'}, 
                        facet_col="Pclass", facet_col_wrap=3,facet_col_spacing=0.11)

        fig.update_layout(legend=dict(orientation='h', yanchor='top', y=1.25, xanchor='center', x=1),
                            xaxis_title='', yaxis_title='Cantidad')
        return fig
    # * Mostrar el gráfico
    st.plotly_chart(generarGrafico(), use_container_width=True) 


    # Ordenar los datos por sexo y supervivencia
    sorted_df = df.sort_values(by=['Sex', 'Survived'])

    # Crear el gráfico de bigotes para la distribución de edades de los pasajeros por sexo y supervivencia
    fig1 = go.Figure(go.Box(x=sorted_df['Sex'], y=sorted_df['Age'], marker_color='blue', name='Embarcados'))

    # Ocultar los puntos
    fig1.update_traces(boxpoints=False)

    # Crear el gráfico de bigotes para la distribución de edades de los supervivientes por sexo
    fig2 = go.Figure(go.Box(x=sorted_df[sorted_df['Survived'] == 1]['Sex'], y=sorted_df[sorted_df['Survived'] == 1]['Age'], 
                            marker_color='green', name='Supervivientes'))

    # Ocultar los puntos
    fig2.update_traces(boxpoints=False)

    # Crear el gráfico de bigotes para la distribución de edades de los fallecidos por sexo
    fig3 = go.Figure(go.Box(x=sorted_df[sorted_df['Survived'] == 0]['Sex'], y=sorted_df[sorted_df['Survived'] == 0]['Age'], 
                            marker_color='red', name='Fallecidos'))

    # Ocultar los puntos
    fig3.update_traces(boxpoints=False)

    # Configurar el rango del eje y en el caso de los fallecidos
    fig3.update_yaxes(range=[0, 80])

    # Configurar subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Embarcados', 'Supervivientes', 'Fallecidos'))

    # Añadir los gráficos a los subplots
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    fig.add_trace(fig3.data[0], row=1, col=3)

    # Actualizar el diseño de los subplots
    fig.update_layout(showlegend=True)

    #* Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True) 
with tabEdades:
    fig = px.histogram(df, 
                x='Age',
                color='Survived',
                title="Distribución de Supervivencia por Edad",
                color_discrete_map={0: "red", 1: "green"},
                marginal="rug",  # Agregar histogramas marginales
                opacity=0.7,  # Reducir la opacidad para visualizar la superposición
                histnorm='probability density'  # Normalizar el histograma para comparar distribuciones
                )

    # Actualizar las etiquetas de las leyendas
    fig.update_traces(name="Fallecidos", selector={"legendgroup": "0"})
    fig.update_traces(name="Supervivientes", selector={"legendgroup": "1"})

    # Actualizar el diseño del gráfico para superponer los histogramas y ajustar el ancho de las barras

    fig.update_layout(barmode="overlay", showlegend=True)  # Superponer los histogramas y mostrar la leyenda

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True) 
with tabNubes:
    row1 = st.columns(2)
    row2 = st.columns(2)

    # Agregar contenido a la primera columna
    with row1[0]:
        st.markdown("<h3 style='text-align: center;'>WordCloud Supervivientes</h3>", unsafe_allow_html=True)
        st.image(wordcloud_supervivientes,use_column_width=True)

    # Agregar contenido a la segunda columna
    with row1[1]:
        st.markdown("<h3 style='text-align: center;'>WordCloud Supervivientes- Filtrado</h3>", unsafe_allow_html=True)
        supervivientes_filtrado = filtered_df[filtered_df['Survived'] == 1]['Apellido']
        wordcloud_supervivientes_filtrado = (WordCloud(width=800, height=400,background_color=background_color,
                                                    mask=mascara,colormap='twilight',random_state=random_state).generate(' '.join(supervivientes_filtrado))).to_array()
        st.image(wordcloud_supervivientes_filtrado, use_column_width=True)

    with row2[0]:  
        st.markdown("<h3 style='text-align: center;'>WordCloud Fallecidos</h3>", unsafe_allow_html=True)
        st.image(wordcloud_fallecidos, use_column_width=True)

    # Agregar contenido a la segunda columna
    with row2[1]:
        st.markdown("<h3 style='text-align: center;'>WordCloud Fallecidos - Filtrado</h3>", unsafe_allow_html=True)
        fallecidos_filtrado = filtered_df[filtered_df['Survived'] == 0]['Apellido']
        wordcloud_fallecidos_filtrado = (WordCloud( width=800, height=400,background_color=background_color,
                                                mask=mascara,colormap='turbo',random_state=random_state).generate(' '.join(fallecidos_filtrado))).to_array()
        st.image(wordcloud_fallecidos_filtrado,use_column_width=True)
