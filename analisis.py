import dash
from dash import dash_table, dcc, html
import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
from scipy.stats import pearsonr, chi2_contingency, ttest_ind, levene
from scipy.stats import chi2_contingency
import io
import base64
import statsmodels.api as sm
import plotly.graph_objects as go 
from itertools import combinations
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu, kruskal
import math
from io import BytesIO
from scipy.stats import kstest, norm
import requests
from scipy.stats import anderson




# Inicializar la aplicación Dash
app = dash.Dash(__name__)

#descriptiva ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

df = pd.read_csv('data.csv')
# Definir las variables para los gráficos
binary_vars = [
    'alfabetismo', 'inasistencia_escolar', 'rezago_escolar', 'atencion_integral',
    'trabajo_infantil', 'empleo_formal', 'desempleo_larga_duracion',
    'barreras_acceso_salud', 'aseguramiento_salud', 'hacinamiento', 'POBRE'
]

multi_cat_vars = ['P8530', 'P8526', 'P1075']




numerical_vars = ['PERSONAS', 'IPM']
stats = df[numerical_vars].describe().transpose().reset_index()
stats.rename(columns={'index': 'Variable'}, inplace=True)

numerical_table = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in stats.columns],
    data=stats.to_dict('records'),
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center'},
    style_header={'fontWeight': 'bold'}
)


var_labels = {
    'P8530': 'Fuente de agua para preparación de alimentos',
    'P8526': 'Tipo de servicio sanitario',
    'P1075': 'Conexión a Internet',
}


# inferencias ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Variables para análisis
numerical_vars1 = ['IPM', 'PERSONAS']
numerical_vars2 = ['IPM', 'P5010']
numerical_vars3 = ['PERSONAS', 'P5010']
vcatg = ['alfabetismo', 'inasistencia_escolar']  # Solo como ejemplo

# Inicializar los resultados
pearson_results = []
chi2_results = []
ttest_results = []
levene_results = []

# Correlación de Pearson
for i in range(len(numerical_vars1)):
    for j in range(i + 1, len(numerical_vars1)):
        var1 = numerical_vars1[i]
        var2 = numerical_vars1[j] 
        pearson_coef, pearson_p = pearsonr(df[var1], df[var2])
        pearson_results.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Coeficiente': round(pearson_coef, 3),
            'P-valor': round(pearson_p, 5)
        })
for i in range(len(numerical_vars2)):
    for j in range(i + 1, len(numerical_vars2)):
        var1 = numerical_vars2[i]
        var2 = numerical_vars2[j] 
        pearson_coef, pearson_p = pearsonr(df[var1], df[var2])
        pearson_results.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Coeficiente': round(pearson_coef, 3),
            'P-valor': round(pearson_p, 5)
        })

for i in range(len(numerical_vars3)):
    for j in range(i + 1, len(numerical_vars3)):
        var1 = numerical_vars3[i]
        var2 = numerical_vars3[j]
        pearson_coef, pearson_p = pearsonr(df[var1], df[var2])
        pearson_results.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Coeficiente': round(pearson_coef, 3),
            'P-valor': round(pearson_p, 5)
        })

# Lista de variables categóricas
vcatg = ['alfabetismo', 'inasistencia_escolar', 'rezago_escolar', 'atencion_integral',
         'trabajo_infantil', 'empleo_formal', 'desempleo_larga_duracion', 'barreras_acceso_salud',
         'aseguramiento_salud', 'hacinamiento', 'POBRE', 'paredes', 'pisos', 'alcantarillado', 'acueducto']

# Crear un diccionario de mapeo para mostrar sin guiones bajos en el Dropdown
display_vcatg = {
    'alfabetismo': 'Alfabetismo',
    'inasistencia_escolar': 'Inasistencia Escolar',
    'rezago_escolar': 'Rezago Escolar',
    'atencion_integral': 'Atención Integral',
    'trabajo_infantil': 'Trabajo Infantil',
    'empleo_formal': 'Empleo Formal',
    'desempleo_larga_duracion': 'Desempleo a Larga Duración',
    'barreras_acceso_salud': 'Barreras de Acceso a Salud',
    'aseguramiento_salud': 'Aseguramiento en Salud',
    'hacinamiento': 'Hacinamiento',
    'POBRE': 'Pobreza',
    'paredes': 'Paredes',
    'pisos': 'Pisos',
    'alcantarillado': 'Alcantarillado',
    'acueducto': 'Acueducto'
}

pairs = combinations(vcatg, 2)
dropdown_options = [{'label': f'{display_vcatg[var1]} vs {display_vcatg[var2]}', 'value': f'{var1}|{var2}'} for var1, var2 in pairs]



levene_results = []  # Lista donde se almacenan los resultados
for var in vcatg:
    categorias = df[var].dropna().unique()
    if len(categorias) == 2:
        grupo_1 = df[df[var] == categorias[0]]['IPM'].dropna()
        grupo_2 = df[df[var] == categorias[1]]['IPM'].dropna()
        stat_levene, p_levene = levene(grupo_1, grupo_2)
        
        # Determinar homocedasticidad basado en el P-valor
        homocedasticidad = 'Sí' if p_levene > 0.05 else 'No'

        # Agregar resultados con la nueva columna
        levene_results.append({
            'Variable': var,
            'Categoría 1': categorias[0],
            'Categoría 2': categorias[1],
            'Levene Statistic': round(stat_levene, 2),
            'P-valor': round(p_levene, 5),
            'Homocedasticidad': homocedasticidad  # Nueva columna
        })


# Prueba de normalidad
normalidad_results = []

for var in vcatg:
    categorias = df[var].dropna().unique()  # Obtener categorías únicas, ignorando valores nulos
    if len(categorias) == 2:  # Asegurar que hay exactamente 2 categorías
        grupo_1 = df[df[var] == categorias[0]]['IPM'].dropna()
        grupo_2 = df[df[var] == categorias[1]]['IPM'].dropna()
        
        # Prueba de Kolmogorov-Smirnov
        stat_1, p_normal_1 = kstest(grupo_1, 'norm', args=(grupo_1.mean(), grupo_1.std()))
        stat_2, p_normal_2 = kstest(grupo_2, 'norm', args=(grupo_2.mean(), grupo_2.std()))
        
        # Evaluar normalidad basado en el P-valor
        normal_1 = 'Sí' if p_normal_1 > 0.05 else 'No'
        normal_2 = 'Sí' if p_normal_2 > 0.05 else 'No'

        # Agregar resultados a la tabla
        normalidad_results.append({
            'Variable': var,
            'Categoría': categorias[0],
            'KS Statistic': round(stat_1, 2),
            'P-valor': round(p_normal_1, 5),
            'Normalidad': normal_1
        })
        normalidad_results.append({
            'Variable': var,
            'Categoría': categorias[1],
            'KS Statistic': round(stat_2, 2),
            'P-valor': round(p_normal_2, 5),
            'Normalidad': normal_2
        })


# Lista donde se almacenarán los resultados de las pruebas
mediana_results = []

for var in vcatg:
    categorias = df[var].dropna().unique()
    if len(categorias) == 2:
        # Dos grupos, realizamos la prueba de Mann-Whitney U
        grupo_1 = df[df[var] == categorias[0]]['IPM'].dropna()
        grupo_2 = df[df[var] == categorias[1]]['IPM'].dropna()

        stat, p_value = mannwhitneyu(grupo_1, grupo_2, alternative='two-sided')
        
        # Determinar si hay diferencia significativa basado en el P-valor
        diferencia_significativa = 'Sí' if p_value <= 0.05 else 'No'
        
        # Agregar los resultados
        mediana_results.append({
            'Variable': var,
            'Categoría 1': categorias[0],
            'Categoría 2': categorias[1],
            'Estadístico U': round(stat, 2),
            'P-valor': round(p_value, 5),
            'Diferencia Significativa': diferencia_significativa,
            'Conclusión': 'Las medianas son diferentes' if diferencia_significativa == 'Sí' else 'No se detecta diferencia en las medianas'
        })
    
    else:
        # Más de dos categorías, realizamos la prueba de Kruskal-Wallis
        grupos = [df[df[var] == cat]['IPM'].dropna() for cat in categorias]
        
        stat, p_value = kruskal(*grupos)
        
        # Determinar si hay diferencia significativa basado en el P-valor
        diferencia_significativa = 'Sí' if p_value <= 0.05 else 'No'
        
        # Agregar los resultados
        mediana_results.append({
            'Variable': var,
            'Categorías': ', '.join(categorias),
            'Estadístico H': round(stat, 2),
            'P-valor': round(p_value, 5),
            'Diferencia Significativa': diferencia_significativa,
            'Conclusión': 'Las medianas son diferentes' if diferencia_significativa == 'Sí' else 'No se detecta diferencia en las medianas'
        })

# Convertimos los resultados en un DataFrame para mostrarlo en la tabla
mediana_results_df = pd.DataFrame(mediana_results)


# Convertir las columnas a categorías si es necesario




# Layout de la aplicación (Página de inicio)
app.layout = html.Div(style={'fontFamily': 'Arial', 'backgroundColor': '#ADD8E6', 'padding': '20px'}, children=[
    
    # Título principal
    html.H1('Dashboard de Análisis de IPM', style={'textAlign': 'center', 'color': '#003366'}),

    # Cargar el logo desde la carpeta 'assets'
    html.Div(
        html.Img(src='https://talentotech.gov.co/849/channels-748_logo_talentotech.png', style={'display': 'block', 'margin': '0 auto', 'width': '300px'}),  # Coma agregada aquí
        style={'textAlign': 'center', 'marginTop': '20px'}
    ),
    
    # Nombres de los integrantes
    html.Div(
        html.H2('Integrantes:', style={'textAlign': 'center', 'color': '#333333'}),
        style={'textAlign': 'center', 'marginTop': '20px'}
    ),
    html.Div(
        html.Ul(children=[
            html.Li('Juan Camilo MOrales Orozco'),
            html.Li('Esteban Jafeth Florez Berrio'),
            html.Li('Maria Angel Ricardo Sierra'),
            html.Li('Oriana Margarita Arguelles Fang'),
            html.Li('Keiver Jose Alvarez Ospino'),
            html.Li('Jose Rafael Barrios Contreras'),
        ], style={'textAlign': 'center', 'color': '#333333'}),
    ),
    
    # Descripción del proyecto
    html.Div(
        html.H3('Descripción del Proyecto:', style={'textAlign': 'center', 'color': '#333333'}),
        style={'textAlign': 'center', 'marginTop': '30px'}
    ),
    html.Div(
        html.P('Este proyecto tiene como objetivo aplicar técnicas avanzadas de analítica de datos para optimizar los procesos en los sectores de educación'
        ' y salud en los departamentos de Bolívar, Córdoba, Sucre y San Andrés. El análisis de datos como herramienta estratégica nos permite identificar patrones, predecir tendencias y mejorar la toma de decisiones en entornos complejos, especialmente en áreas donde el acceso a servicios esenciales está severamente limitado. Al analizar indicadores clave e identificar relaciones entre variables, se busca obtener una comprensión más profunda de la dinámica de estos sectores clave, sentando las bases para intervenciones efectivas que mejoren la calidad de vida y reduzcan la desigualdad en estas comunidades.'),
        style={'textAlign': 'center', 'color': '#333333', 'marginTop': '20px', 'borderBottom': '2px solid #333333', 'paddingBottom': '20px'}
    ),

    
    # Presentación de la Base de Datos
    html.Div(
        html.H1('Presentación de la Base de Datos', style={'textAlign': 'center', 'color': '#003366'}),
        style={'textAlign': 'center', 'marginTop': '20px'}
    ),
    
    # Mostrar la tabla de las primeras filas de la base de datos
    dash_table.DataTable(
        id='tabla-base-datos',
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.head(10).to_dict('records'),
        style_table={'margin': '0 auto', 'width': '80%', 'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontSize': '14px', 'fontFamily': 'Arial'},
    ),

    html.Div(
    html.H3('Tipos de Datos de las Variables', style={'textAlign': 'center', 'color': '#333333'}),
    style={'textAlign': 'center', 'marginTop': '30px'}
    ),

    # Crear una tabla para mostrar los tipos de datos
    dash_table.DataTable(
        id='tipos-datos',
        columns=[
            {"name": "Variable", "id": "variable"},
            {"name": "Tipo de Datos", "id": "tipo_dato"}
        ],
        data=[{"variable": col, "tipo_dato": str(df[col].dtype)} for col in df.columns],
        style_table={
            'margin': '0 auto', 
            'width': '80%', 
            'overflowX': 'auto', 
            'border': '1px solid #ccc',  # Borde alrededor de la tabla
            'borderRadius': '10px',  # Bordes redondeados
            'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',  # Sombra para resaltar la tabla
            'backgroundColor': '#ffffff'  # Fondo blanco para la tabla
        },
        style_header={
            'backgroundColor': '#f1f1f1',  # Color de fondo de los encabezados
            'fontWeight': 'bold',  # Negrita para los encabezados
            'textAlign': 'center', 
            'color': '#333333',  # Color de texto de los encabezados
            'padding': '10px'  # Espaciado en los encabezados
        },
        style_cell={
            'textAlign': 'center', 
            'padding': '10px', 
            'fontSize': '14px', 
            'fontFamily': 'Arial', 
            'color': '#333333',
            'borderBottom': '1px solid #ddd',  # Línea de borde para las celdas
        },
        style_data={
            'backgroundColor': '#f9f9f9',  # Color de fondo para las filas de la tabla
            'borderBottom': '1px solid #ddd'  # Borde inferior en las filas
        },
        style_data_conditional=[{
            'if': {'row_index': 'odd'},  # Estilo para filas impares
            'backgroundColor': '#f1f1f1',  # Color de fondo diferente para filas impares
        }],
    ),

    


    # Mostrar si hay datos faltantes en cada columna
    html.Div(
        html.H3('Datos Faltantes en las Variables', style={'textAlign': 'center', 'color': '#333333'}),
        style={'textAlign': 'center', 'marginTop': '30px'}
    ),
    
    # Crear una tabla para mostrar los datos faltantes
    dash_table.DataTable(
        id='datos-faltantes',
        columns=[
            {"name": "Variable", "id": "variable"},
            {"name": "Datos Faltantes", "id": "datos_faltantes"}
        ],
        data=[{"variable": col, "datos_faltantes": df[col].isnull().sum()} for col in df.columns],
        style_table={
            'margin': '0 auto', 
            'width': '80%', 
            'overflowX': 'auto', 
            'border': '1px solid #ccc',  # Borde alrededor de la tabla
            'borderRadius': '10px',  # Bordes redondeados
            'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',  # Sombra para resaltar la tabla
            'backgroundColor': '#ffffff'  # Fondo blanco para la tabla
        },
        style_header={
            'backgroundColor': '#f1f1f1',  # Color de fondo de los encabezados
            'fontWeight': 'bold',  # Negrita para los encabezados
            'textAlign': 'center', 
            'color': '#333333',  # Color de texto de los encabezados
            'padding': '10px'  # Espaciado en los encabezados
        },
        style_cell={
            'textAlign': 'center', 
            'padding': '10px', 
            'fontSize': '14px', 
            'fontFamily': 'Arial', 
            'color': '#333333',
            'borderBottom': '1px solid #ddd',  # Línea de borde para las celdas
        },
        style_data={
            'backgroundColor': '#f9f9f9',  # Color de fondo para las filas de la tabla
            'borderBottom': '1px solid #ddd'  # Borde inferior en las filas
        },
        style_data_conditional=[{
            'if': {'row_index': 'odd'},  # Estilo para filas impares
            'backgroundColor': '#f1f1f1',  # Color de fondo diferente para filas impares
        }],
    ),

    html.Div(
        html.P('El Índice de Pobreza Multidimensional (IPM) de Colombia es una encuesta estadística llevada a cabo por el DANE con el objetivo de medir las condiciones de pobreza a nivel de hogares en múltiples dimensiones. Fue creado el 18 de octubre de 2024 y se actualiza regularmente para reflejar las condiciones de vida y privaciones que enfrentan los hogares colombianos. Esta encuesta permite el diseño, monitoreo y evaluación de políticas públicas en pro de mejorar las condiciones de vida en el país. '),
        style={'textAlign': 'center', 'color': '#333333', 'marginTop': '20px', 'borderBottom': '2px solid #333333', 'paddingBottom': '20px'}
    ),

    html.H1("Análisis Unidimensional del Proyecto", style={'textAlign': 'center', 'color': '#003366'}),

    # Tabla de estadísticas descriptivas para variables numéricas
    html.Div([
        html.H3("Estadísticas Descriptivas - Variables Numéricas"),
        html.Div(numerical_table, style={'marginBottom': '20px'})
    ], style={'padding': 20}),

  
    
    # Histogramas y boxplots para IPM y PERSONAS
    html.Div([
        html.H3("Distribución de IPM y PERSONAS"),
        html.Div([
            dcc.Graph(
                id='IPM-histogram',
                figure=px.histogram(df, x='IPM', nbins=15, title="Distribución del IPM")
            ),
            dcc.Graph(
                id='IPM-boxplot',
                figure=px.box(df, y='IPM', title="Boxplot del IPM")
            )
        ], style={'display': 'flex', 'flexDirection': 'row'}),
        html.Div([
            dcc.Graph(
                id='PERSONAS-histogram',
                figure=px.histogram(df, x='PERSONAS', nbins=10, title="Distribución de PERSONAS por hogar")
            ),
            dcc.Graph(
                id='PERSONAS-boxplot',
                figure=px.box(df, y='PERSONAS', title="Boxplot de PERSONAS por hogar")
            )
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ], style={'padding': 20}),

    # Gráfico de las variables categóricas con múltiples respuestas
    html.Div([
        html.H3("Distribución de Variables con Respuestas Múltiples"),
        *[
            html.Div([
                dcc.Graph(
                    id=f'{var}-graph',
                    figure={
                        'data': [
                            {
                                'x': df[var].value_counts().index,  # Categorías únicas
                                'y': df[var].value_counts().values,  # Frecuencias
                                'type': 'bar',
                                'name': var_labels[var]
                            }
                        ],
                        'layout': {
                            'title': f'Distribución de {var_labels[var]}',
                            'xaxis': {'title': var_labels[var]},
                            'yaxis': {'title': 'Frecuencia'}
                        }
                    }
                )
            ], style={'padding': 20})
            for var in multi_cat_vars
        ]
    ], style={'padding': 20}),
 
    
    # Gráfico de las variables binarias
    html.Div([
        html.H3("Distribución de Variables Binarias"),
        dcc.Graph(
            id='binary-vars-graph',
            figure={
                'data': [
                    {
                        'x': binary_vars,
                        'y': [df[var].value_counts().get('Sí', 0) for var in binary_vars],  # Contar "sí"
                        'type': 'bar',
                        'name': 'Privación'
                    },
                    {
                        'x': binary_vars,
                        'y': [df[var].value_counts().get('No', 0) for var in binary_vars],  # Contar "no"
                        'type': 'bar',
                        'name': 'No Privación'
                    }
                ],
                'layout': {
                    'title': 'Distribución de Privación en Variables Binarias'
                }
            }
        )
    ], style={'padding': 20}),



    
    # Gráfico de barras apiladas para la pobreza por departamento
    html.Div([
        html.H3("Proporción de Pobreza por Departamento"),
        dcc.Graph(
            id='poverty-department-graph',
            figure={
                'data': [
                    {
                        'x': df['DEPARTAMENTO'].unique(),
                        'y': df.groupby(['DEPARTAMENTO', 'POBRE']).size().unstack().div(df.groupby('DEPARTAMENTO').size(), axis=0).get('Sí', 0),  # Para "Pobre"
                        'type': 'bar',
                        'name': 'Pobre'
                    },
                    {
                        'x': df['DEPARTAMENTO'].unique(),
                        'y': df.groupby(['DEPARTAMENTO', 'POBRE']).size().unstack().div(df.groupby('DEPARTAMENTO').size(), axis=0).get('No', 0),  # Para "No Pobre"
                        'type': 'bar',
                        'name': 'No Pobre'
                    }
                ],
                'layout': {
                    'title': 'Proporción de Pobreza por Departamento',
                    'barmode': 'stacked'
                }
            }
        )
    ], style={'padding': 20}),

    html.Div(
        html.P('El análisis unidimensional nos permitió comprender las características generales de las variables clave del proyecto, como el Índice de Pobreza Multidimensional (IPM) y el número de personas por hogar. Este enfoque inicial nos brinda una base sólida para explorar patrones y tendencias en los datos, asegurando una mejor interpretación y modelado posterior.'),
        style={'textAlign': 'center', 'color': '#333333', 'marginTop': '20px', 'borderBottom': '2px solid #333333', 'paddingBottom': '20px'}
    ),
  
    

    html.H1("Análisis Bidimencional del Proyecto", style={'textAlign': 'center', 'color': '#003366'}),

    # Gráfico Boxplot para IPM por Departamento
    html.Div([
        html.H3("Distribución del IPM por Departamento"),
        dcc.Graph(
            id='ipm-department-boxplot',
            figure={
                'data': [
                    {
                        'x': df['DEPARTAMENTO'],
                        'y': df['IPM'],
                        'type': 'box',
                        'name': 'IPM',
                        'marker': {'color': 'lightblue'}
                    }
                ],
                'layout': {
                    'title': 'Distribución del IPM por Departamento',
                    'xaxis': {'title': 'Departamento', 'tickangle': 45},
                    'yaxis': {'title': 'Índice de Pobreza Multidimensional (IPM)'}
                }
            }
        )
    ], style={'padding': 20}),

    # Gráficos Boxplot para variables categóricas
    html.Div([
        html.H3("Análisis Bivariado: IPM vs Variables Categóricas"),
        *[
            html.Div([
                dcc.Graph(
                    id=f'boxplot-{var}',
                    figure={
                        'data': [
                            {
                                'x': df[var],
                                'y': df['IPM'],
                                'type': 'box',
                                'name': var,
                                'marker': {'color': 'mediumseagreen'}
                            }
                        ],
                        'layout': {
                            'title': f'Distribución del IPM por {var.capitalize()}',
                            'xaxis': {'title': var.capitalize()},
                            'yaxis': {'title': 'Índice de Pobreza Multidimensional (IPM)'}
                        }
                    }
                )
            ], style={'padding': 20})
            for var in binary_vars
        ]
    ], style={'padding': 20}),



    html.Div(
        html.P('El análisis bidimensional nos permitió explorar las relaciones entre variables como el IPM y las características de los hogares. Estas interacciones son cruciales para identificar posibles factores asociados con la pobreza y desarrollar estrategias basadas en evidencia. Además, proporciona una perspectiva más profunda que complementa el análisis unidimensional.'),
        style={'textAlign': 'center', 'color': '#333333', 'marginTop': '20px', 'borderBottom': '2px solid #333333', 'paddingBottom': '20px'}
    ),

    html.H1("Análisis Inferencial", style={'textAlign': 'center', 'color': '#003366'}),



    
    # Correlación de Pearson
    html.Div([
        html.H3("Correlación de Pearson", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='pearson-table',
            columns=[
                {'name': 'Variable 1', 'id': 'Variable 1'},
                {'name': 'Variable 2', 'id': 'Variable 2'},
                {'name': 'Coeficiente', 'id': 'Coeficiente'},
                {'name': 'P-valor', 'id': 'P-valor'}
            ],
            data=pearson_results
        )
    ], style={'padding': '20px'}),

    # Gráfico de dispersión entre PERSONAS e IPM
    html.Div([
        html.H3("Correlación entre PERSONAS y IPM"),
        dcc.Graph(
            id='scatter-personas-ipm',
            figure={
                'data': [
                    {
                        'x': df['PERSONAS'],
                        'y': df['IPM'],
                        'mode': 'markers',
                        'marker': {'size': 8, 'color': 'tomato'},
                        'name': 'PERSONAS vs IPM'
                    }
                ],
                'layout': {
                    'title': 'Relación entre el número de PERSONAS e IPM',
                    'xaxis': {'title': 'Número de PERSONAS'},
                    'yaxis': {'title': 'Índice de Pobreza Multidimensional (IPM)'}
                }
            }
        )
    ], style={'padding': 20}),


    # Gráfico de regresión lineal entre PERSONAS e IPM
    html.Div([
        html.H3("Relación entre PERSONAS y IPM (Regresión Lineal)"),
        dcc.Graph(
            id='regression-personas-ipm',
            figure={
                'data': [
                    {
                        'x': df['PERSONAS'],
                        'y': df['IPM'],
                        'mode': 'markers',
                        'name': 'Datos',
                        'marker': {'size': 8, 'color': 'tomato'}
                    },
                    {
                        'x': df['PERSONAS'],
                        'y': sm.OLS(df['IPM'], sm.add_constant(df['PERSONAS'])).fit().predict(sm.add_constant(df['PERSONAS'])),
                        'mode': 'lines',
                        'name': 'Regresión Lineal',
                        'line': {'color': 'blue', 'width': 2}
                    }
                ],
                'layout': {
                    'title': 'Relación entre el número de PERSONAS e IPM',
                    'xaxis': {'title': 'Número de PERSONAS'},
                    'yaxis': {'title': 'Índice de Pobreza Multidimensional (IPM)'},
                    'showlegend': True
                }
            }
        )
    ], style={'padding': 20}),

    # Gráfico de dispersión entre IPM y P5010
    html.Div([
        html.H3("Correlación entre IPM y P5010"),
        dcc.Graph(
            id='scatter-ipm-p5010',
            figure={
                'data': [
                    {
                        'x': df['IPM'],
                        'y': df['P5010'],
                        'mode': 'markers',
                        'marker': {'size': 8, 'color': 'blue'},
                        'name': 'IPM vs P5010'
                    }
                ],
                'layout': {
                    'title': 'Relación entre el Índice de Pobreza Multidimensional (IPM) y el Número de Cuartos (P5010)',
                    'xaxis': {'title': 'Índice de Pobreza Multidimensional (IPM)'},
                    'yaxis': {'title': 'Número de Cuartos (P5010)'}
                }
            }
        )
    ], style={'padding': 20}),


    # Gráfico de dispersión entre PERSONAS y P5010
    html.Div([
        html.H3("Correlación entre PERSONAS y P5010"),
        dcc.Graph(
            id='scatter-personas-p5010',
            figure={
                'data': [
                    {
                        'x': df['PERSONAS'],
                        'y': df['P5010'],
                        'mode': 'markers',
                        'marker': {'size': 8, 'color': 'green'},
                        'name': 'PERSONAS vs P5010'
                    }
                ],
                'layout': {
                    'title': 'Relación entre el número de PERSONAS y el Número de Cuartos (P5010)',
                    'xaxis': {'title': 'Número de PERSONAS'},
                    'yaxis': {'title': 'Número de Cuartos (P5010)'}
                }
            }
        )
    ], style={'padding': 20}),





    html.H1("Resultados de las pruebas de Chi-cuadrado entre variables categóricas"),
    dcc.Dropdown(
        id='chi2-dropdown',
        options=[
            {'label': f'{display_vcatg[var1]} vs {display_vcatg[var2]}', 'value': f'{var1}|{var2}'}
            for var1, var2 in combinations(vcatg, 2)
        ],
        value=f'{vcatg[0]}|{vcatg[1]}',  # Valor inicial
        multi=False,
        placeholder="Selecciona un par de variables"
    ),
    html.Div(id='chi2-table', style={'margin-top': '20px'}),  # Espaciado entre elementos
    html.Div(id='chi2-result', style={'margin-top': '20px'}),
    dcc.Graph(id='chi2-plot', style={'margin-top': '20px'}),

    html.Hr(),  # Separador visual entre secciones

    # Sección: Análisis de normalidad y medianas
    html.H1("Análisis de Normalidad y Medianas por Variable"),
    dcc.Dropdown(
        id='normality-dropdown',
        options=[{'label': display_vcatg[var], 'value': var} for var in vcatg],
        value=vcatg[0],
        multi=False,
        placeholder="Selecciona una variable"
    ),
    html.Div(id='analysis-table', style={'margin-top': '20px'}),
    html.Div(id='analysis-result', style={'margin-top': '20px'}),

    # Prueba de Normalidad
    html.Div([
        html.H3("Prueba de Normalidad (Kolmogorov-Smirnov)", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='normalidad-table',
            columns=[
                {'name': 'Variable', 'id': 'Variable'},
                {'name': 'Categoría', 'id': 'Categoría'},
                {'name': 'KS Statistic', 'id': 'KS Statistic'},
                {'name': 'P-valor', 'id': 'P-valor'},
                {'name': 'Normalidad', 'id': 'Normalidad'}  # Columna indicando si la distribución es normal
            ],
            data=normalidad_results,  # Datos con los resultados de la prueba de normalidad
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
    ], style={'padding': '20px'}),

    # Prueba de Homocedasticidad
    html.Div([
        html.H3("Prueba de Homocedasticidad (Levene)", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='levene-table',
            columns=[
                {'name': 'Variable', 'id': 'Variable'},
                {'name': 'Categoría 1', 'id': 'Categoría 1'},
                {'name': 'Categoría 2', 'id': 'Categoría 2'},
                {'name': 'Levene Statistic', 'id': 'Levene Statistic'},
                {'name': 'P-valor', 'id': 'P-valor'},
                {'name': 'Homocedasticidad', 'id': 'Homocedasticidad'}  # Nueva columna
            ],
            data=levene_results  # Datos actualizados con la columna "Homocedasticidad"
        )
    ], style={'padding': '20px'}),





    # Mostrar la tabla en tu layout
    html.Div([
        html.H3("Prueba de Mediana", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='mediana-table',
            columns=[
                {'name': 'Variable', 'id': 'Variable'},
                {'name': 'Categoría 1', 'id': 'Categoría 1'},
                {'name': 'Categoría 2', 'id': 'Categoría 2'},
                {'name': 'Estadístico U', 'id': 'Estadístico U'},
                {'name': 'Estadístico H', 'id': 'Estadístico H'},
                {'name': 'P-valor', 'id': 'P-valor'},
                {'name': 'Diferencia Significativa', 'id': 'Diferencia Significativa'},
                {'name': 'Conclusión', 'id': 'Conclusión'}
            ],
            data=mediana_results_df.to_dict('records')  # Convertimos a diccionario para el DataTable
        )
    ], style={'padding': '20px'})

])

@app.callback(
    [Output('chi2-table', 'children'),
     Output('chi2-result', 'children')],
    [Input('chi2-dropdown', 'value')]
)
def update_chi2_result(selected_pair):
    print(f"Seleccionado: {selected_pair}")  # Para depuración

    if selected_pair:
        try:
            # Dividir las variables usando el delimitador único '|'
            var1, var2 = selected_pair.split('|')
            
            # Asegurarse de que las variables se seleccionan correctamente
            print(f"Variables seleccionadas: {var1}, {var2}")

            # Crear la tabla de contingencia entre las dos variables
            contingency_table = pd.crosstab(df[var1], df[var2])
            print(f"Tabla de contingencia entre {var1} y {var2}:\n{contingency_table}")

            # Realizar la prueba de Chi-cuadrado
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Crear el contenido de la tabla de contingencia como HTML
            table_html = html.Table([
                html.Thead(html.Tr([html.Th(col) for col in contingency_table.columns])),
                html.Tbody([
                    html.Tr([html.Td(contingency_table.iloc[i, j]) for j in range(len(contingency_table.columns))])
                    for i in range(len(contingency_table))
                ])
            ])

            # Crear el resultado de la prueba Chi-cuadrado como HTML
            result_html = html.Div([
                html.H3(f"Resultados de la prueba Chi-cuadrado para {display_vcatg[var1]} y {display_vcatg[var2]}:"),
                html.P(f"  - Estadístico Chi-cuadrado: {chi2:.2f}"),
                html.P(f"  - p-valor: {p:.5e}"),
                html.P(f"  - Grados de libertad: {dof}"),
                html.P("  - Conclusión: " + ("Existe evidencia para rechazar la hipótesis nula. Las variables son dependientes." if p < 0.05 else "No hay evidencia suficiente para rechazar la hipótesis nula. Las variables son independientes."))
            ])

            return table_html, result_html

        except Exception as e:
            print(f"Error al procesar las variables: {e}")
            return html.Div("Error en las variables seleccionadas."), html.Div("No se pudo realizar la prueba.")
    else:
        return html.Div("Por favor selecciona una combinación de variables."), html.Div("No se pudo realizar la prueba.")


@app.callback(
    Output('chi2-plot', 'figure'),
    [Input('chi2-dropdown', 'value')]
)
def update_chi2_plot(selected_pair):
    # Tu lógica aquí

    if selected_pair:
        try:
            # Dividir las variables seleccionadas
            var1, var2 = selected_pair.split('|')
            
            # Crear tabla de contingencia
            contingency_table = pd.crosstab(df[var1], df[var2])
            
            # Generar gráfico de barras apiladas
            fig = px.bar(
                contingency_table,
                barmode='stack',
                title=f"Distribución de {display_vcatg[var1]} vs {display_vcatg[var2]}",
                labels={'value': 'Frecuencia', 'index': display_vcatg[var1]},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(xaxis_title=display_vcatg[var1], yaxis_title="Frecuencia")
            
            return fig
        except Exception as e:
            print(f"Error al generar el gráfico: {e}")
            return px.Figure()
    else:
        # Gráfico vacío si no hay selección
        return px.Figure()


@app.callback(
    [Output('analysis-table', 'children'),
     Output('analysis-result', 'children')],
    [Input('normality-dropdown', 'value')]
)
def update_analysis_result(selected_variable):
    if selected_variable:
        try:
            # Filtrar los datos según la variable seleccionada
            group_si = df[df[selected_variable] == 'Sí']['IPM']
            group_no = df[df[selected_variable] == 'No']['IPM']

            # Validar que ambos grupos tienen datos suficientes
            if group_si.empty or group_no.empty:
                return (
                    html.Div("Uno de los grupos no tiene suficientes datos para realizar análisis."),
                    html.Div("Por favor selecciona una variable con datos válidos.")
                )

            # Resultados de las pruebas
            results = {}

            # Prueba Anderson-Darling
            anderson_si = anderson(group_si)
            anderson_no = anderson(group_no)

            # Prueba Kolmogorov-Smirnov
            ks_si_stat, ks_si_pvalue = kstest(group_si, 'norm', alternative='two-sided') if not group_si.empty else (None, None)
            ks_no_stat, ks_no_pvalue = kstest(group_no, 'norm', alternative='two-sided') if not group_no.empty else (None, None)

            # Prueba de Levene (para igualdad de varianzas)
            levene_stat, levene_pvalue = levene(group_si, group_no) if not group_si.empty and not group_no.empty else (None, None)

            # Prueba Mann-Whitney U
            mannwhitney_stat, mannwhitney_pvalue = mannwhitneyu(group_si, group_no, alternative='two-sided') if not group_si.empty and not group_no.empty else (None, None)

            # Crear la tabla de resultados
            results_table = html.Table([
                html.Thead(html.Tr([html.Th("Prueba"), html.Th("Grupo 'Sí'"), html.Th("Grupo 'No'"), html.Th("Conclusión")])),
                html.Tbody([
                    html.Tr([
                        html.Td("Anderson-Darling"),
                        html.Td(f"Estadístico: {anderson_si.statistic:.2f}, p-valor: N/A"),
                        html.Td(f"Estadístico: {anderson_no.statistic:.2f}, p-valor: N/A"),
                        html.Td("Normal" if anderson_si.statistic < anderson_si.critical_values[2] and anderson_no.statistic < anderson_no.critical_values[2]
                                else "No normal")
                    ]),
                    html.Tr([
                        html.Td("Kolmogorov-Smirnov"),
                        html.Td(f"p-valor: {ks_si_pvalue:.5f}" if ks_si_pvalue is not None else "No disponible"),
                        html.Td(f"p-valor: {ks_no_pvalue:.5f}" if ks_no_pvalue is not None else "No disponible"),
                        html.Td("Distribución normal" if ks_si_pvalue is not None and ks_si_pvalue > 0.05 and ks_no_pvalue > 0.05 else "No es normal")
                    ]),
                    html.Tr([
                        html.Td("Levene (Varianzas)"),
                        html.Td(f"p-valor: {levene_pvalue:.5f}" if levene_pvalue is not None else "No disponible"),
                        html.Td("N/A"),
                        html.Td("Varianzas iguales" if levene_pvalue is not None and levene_pvalue > 0.05 else "Varianzas diferentes")
                    ]),
                    html.Tr([
                        html.Td("Mann-Whitney U"),
                        html.Td(f"Estadístico: {mannwhitney_stat:.2f}" if mannwhitney_stat is not None else "No disponible"),
                        html.Td(f"p-valor: {mannwhitney_pvalue:.5f}" if mannwhitney_pvalue is not None else "No disponible"),
                        html.Td("Grupos significativamente diferentes" if mannwhitney_pvalue is not None and mannwhitney_pvalue < 0.05 else "Grupos no significativamente diferentes")
                    ])
                ])
            ])

            # Crear la conclusión general
            normality_conclusion = ("Ambos grupos son normales."
                                    if anderson_si.statistic < anderson_si.critical_values[2] and anderson_no.statistic < anderson_no.critical_values[2]
                                    else "Al menos un grupo no es normal.")

            variance_conclusion = ("Las varianzas entre los grupos son iguales."
                                    if levene_pvalue is not None and levene_pvalue > 0.05 else "Las varianzas entre los grupos son diferentes.")

            mannwhitney_conclusion = ("No hay evidencia suficiente para afirmar diferencias significativas entre los grupos."
                                       if mannwhitney_pvalue is not None and mannwhitney_pvalue > 0.05 else "Existen diferencias significativas entre los grupos.")

            overall_conclusion = f"{normality_conclusion} {variance_conclusion} {mannwhitney_conclusion}"

            # Retornar resultados
            return (
                results_table,
                html.Div([
                    html.H3(f"Resultados para la variable: {selected_variable}"),
                    html.P(overall_conclusion)
                ])
            )

        except Exception as e:
            # Manejar errores
            return html.Div("Error en los cálculos."), html.Div(f"Detalles del error: {str(e)}")
    else:
        return (
            html.Div("Por favor selecciona una variable."),
            html.Div()
        )



        
# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

