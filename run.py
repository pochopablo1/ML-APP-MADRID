from flask import Flask, render_template, request, redirect, url_for
import pickle
import joblib
import pandas as pd
from geopy.distance import geodesic
import sys
import os

app = Flask(__name__)

# Diccionario que asocia distritos de Madrid con latitud y longitud
distrito_coords = {
    'Centro': {'latitud': 40.415363, 'longitud': -3.707398},
    'Arganzuela': {'latitud': 40.398068, 'longitud': -3.693734},
    'Retiro': {'latitud': 40.413647, 'longitud': -3.683649},
    'Salamanca': {'latitud': 40.430828, 'longitud': -3.677558},
    'Chamartín': {'latitud': 40.458499, 'longitud': -3.676797},
    'Tetuán': {'latitud': 40.466731, 'longitud': -3.697976},
    'Chamberí': {'latitud': 40.434224, 'longitud': -3.703356},
    'Fuencarral-El Pardo': {'latitud': 40.520389, 'longitud': -3.794518},
    'Moncloa-Aravaca': {'latitud': 40.439079, 'longitud': -3.740588},
    'Latina': {'latitud': 40.402473, 'longitud': -3.741287},
    'Carabanchel': {'latitud': 40.383669, 'longitud': -3.727319},
    'Usera': {'latitud': 40.385735, 'longitud': -3.710303},
    'Puente de Vallecas': {'latitud': 40.393564, 'longitud': -3.657947},
    'Moratalaz': {'latitud': 40.407917, 'longitud': -3.644423},
    'Ciudad Lineal': {'latitud': 40.445808, 'longitud': -3.649825},
    'Hortaleza': {'latitud': 40.474622, 'longitud': -3.642772},
    'Villaverde': {'latitud': 40.345091, 'longitud': -3.713816},
    'Villa de Vallecas': {'latitud': 40.375867, 'longitud': -3.621322},
    'Vicálvaro': {'latitud': 40.397394, 'longitud': -3.608791},
    'San Blas-Canillejas': {'latitud': 40.435982, 'longitud': -3.604611},
    'Barajas': {'latitud': 40.473280, 'longitud': -3.579847}
}

def escalar(df, sc):

    # df 
    # columnas a escalar
    col=['Latitud','Longitud','Huespedes', 'Habitaciones', 'Banos']
    # scaler

    # escalar 
    df_temp=pd.DataFrame(sc.transform(df[col]), columns=col)
    
    df[col]=df_temp[col]

    return df

def escalar_med(df, sc):

    # df 
    # columnas a escalar
    col=['area', 'latitud', 'longitud', 'n_rooms', 'n_baths']
    # scaler

    # escalar 
    df_temp=pd.DataFrame(sc.transform(df[col]), columns=col)
    
    df[col]=df_temp[col]

    return df

import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


sys.path.append(os.path.dirname(os.getcwd()))

#### CARGAR MODELOS CORTA ####

cl0=joblib.load('./templates/app_corta_estancia/models/modelo_cluster_0.pkl')
cl1=joblib.load('./templates/app_corta_estancia/models/modelo_cluster_1.pkl')
cl2=joblib.load('./templates/app_corta_estancia/models/modelo_cluster_2.pkl')

# CARGA MODELOS MEDIA

cl5_med=joblib.load('./templates/app_media_estancia/models/model_med_cluster_5.pkl')
cl4_med=joblib.load('./templates/app_media_estancia/models/model_med_cluster_4.pkl')
cl0_med=joblib.load('./templates/app_media_estancia/models/model_med_cluster_0.pkl')
cl1_med=joblib.load('./templates/app_media_estancia/models/model_med_cluster_1.pkl')

# CLUSTER CORTA

func_kmeans=joblib.load('./templates/app_corta_estancia/models/cluster.pkl')

# CLUSTER MEDIA

func_kmeans_med=joblib.load('./templates/app_media_estancia/models/kmeans.pkl')


# SCALER CORTA

scaler=joblib.load('./templates/app_corta_estancia/models/mod_scal_fit.pkl')

# SCALER MEDIA

scaler_med=joblib.load('./templates/app_media_estancia/models/scaler.pkl')

def cercanos(area, lat, lon, cluster, hab, ba):

    # leer df original
    df=pd.read_csv('./templates/app_media_estancia/data/df_limpio_clusters.csv')

    # Filtrar df original con valores elegidos por el cliente (area, hab, baños) y se añade el cluster

    df=pd.read_csv('./templates/app_media_estancia/data/df_limpio_clusters.csv')

    df_filtrado= df[(df.cluster==cluster) 
                    & (df.area.between(int(area)*0.8, int(area)*1.2))
                    & (df.n_rooms.between(int(hab)-1,int(hab)+1))
                    & (df.n_baths.between(int(ba)-1,int(ba)+1))
                    ]
    

    # df_filtrado = df[(df['area'] == cliente.loc[0,'area']) 
    #                  & (df['latitud'] == cliente.loc[0,'latitud']) 
    #                  & (df['longitud'] == cliente.loc[0,'longitud'])
    #                  & (df['tipology'] == cliente.loc[0,'tipology']) 
    #                  & (df['n_rooms'] == cliente.loc[0,'n_rooms']) 
    #                  & (df['n_baths'] == cliente.loc[0,'n_baths']) 
    #                  & (df['has_garage'] == cliente.loc[0,'has_garage']) 
    #                  & (df['has_pool'] == cliente.loc[0,'has_pool']) 
    #                  & (df['has_elevator'] == cliente.loc[0,'has_elevator']) 
    #                  & (df['is_exterior'] == cliente.loc[0,'is_exterior'])
    #                  ]

    # Sacar distancias entre dos coordenadas y se añaden a distancias[]

    distancias = []
    # Itera las filas del df
    for index, row in df_filtrado.iterrows():
        coord_punto_referencia = (lat, lon)
        coord_fila = (row['latitud'], row['longitud'])
        # distancia entre el punto de referencia y las coordenadas de la fila
        distancia = geodesic(coord_punto_referencia, coord_fila).kilometers
        # distancia a la lista
        distancias.append(distancia)

    # distancias como una nueva columna al DataFrame
    df_filtrado['distancia_al_punto_referencia'] = distancias
    
    # si ha encontrado casas con mismas características

    if len(distancias) > 0: 
        
        # ordeno df
        df_ordenado = df_filtrado.sort_values(by='distancia_al_punto_referencia')  

        return df_ordenado

    else: 
        vacio=pd.DataFrame()
        return vacio



# Ruta de la aplicación de corta estancia
@app.route('/corta_estancia')
def corta_estancia_index():
    return render_template('app_corta_estancia/index.html', distrito_coords=distrito_coords)

# Ruta de la aplicación de media estancia
@app.route('/media_estancia')
def media_estancia_index():
    return render_template('app_media_estancia/index_media.html', distrito_coords=distrito_coords)

# Ruta para la página principal de selección de la aplicación
@app.route('/')
def seleccionar_app():
    return render_template('seleccionar_app.html')

# Ruta para manejar las predicciones de corta estancia
@app.route('/predecir_corta', methods=['POST'])
def predecir_corta():
    resultado_random_forest_corta = None
    try:
        # Obtener valores de entrada del formulario
        
        dist = request.form.get('distrito')

        latitud = distrito_coords[dist]['latitud']
        longitud = distrito_coords[dist]['longitud']

        tipo_encoded = int(request.form.get('tipo_encoded'))
        huespedes = float(request.form.get('huespedes'))
        habitaciones = float(request.form.get('habitaciones'))
        banos = float(request.form.get('banos'))
        cocina = int(request.form.get('cocina', 0))  # 1 si está marcado, 0 si no
        ac = int(request.form.get('ac', 0))
        wifi = int(request.form.get('wifi', 0))
        calefaccion = int(request.form.get('calefaccion', 0))
        ascensor = int(request.form.get('ascensor', 0))
        parking = int(request.form.get('parking', 0))



        lista_valores_cort=[[latitud, longitud,tipo_encoded, huespedes, habitaciones, banos, cocina, ac, wifi, calefaccion, ascensor, parking, ]]
        # Realizar predicciones utilizando el modelo de corta estancia
        # Pasar a DF
        cliente = pd.DataFrame(lista_valores_cort,columns=[['Latitud', 'Longitud', 'Tipo_encoded', 'Huespedes','Habitaciones', 
       'Banos', 'Cocina','AC','Wifi','Calefacción', 'Ascensor','Parking']])
        
        # se escala una copia para no perder datos originales
        escalado=cliente.copy()

        escalado=escalar(escalado, scaler)

        
        # buscar cluster
         
        clstr=int(func_kmeans.predict(escalado))
        
        # aplicar modelo      

        if clstr==0: 
            pred=cl0.predict(escalado)


        elif clstr==1: 
            pred=cl1.predict(escalado)

        
        elif clstr==2: 
            pred=cl2.predict(escalado)  

        resultado_random_forest_corta = pred
        resultado_random_forest_corta = resultado_random_forest_corta[0]  # Convierte el resultado a un valor único
        return render_template('app_corta_estancia/resultado_corta.html',cliente=cliente, distrito=dist, resulpred=resultado_random_forest_corta)

    except Exception as e:
        error_message = f"Error en la predicción: {str(e)}"
        return render_template('error.html', error_message=error_message)
    

# Ruta para manejar las predicciones de media estancia
@app.route('/predecir_media', methods=['POST'])
def predecir_media():
    try:

        #### RECOGER VALORES DEL html ####
        
        # Distrito 
        dist=request.form.get('distrito')
        # print(dist)
        
        # pasar a lat lon con el diccionario
        lat = distrito_coords[dist]['latitud']
        lon = distrito_coords[dist]['longitud']
        
        # Tamaño 30-600 m2
        area=request.form.get('tamaño')

        # Tipo (Piso/Casa) Transformar a 1/0
        t= request.form.get('tipo')
        if t=='Piso':
            tipo=1
        else: 
            tipo=0

        # Nº de habitaciones 0-20 (0 -> Loft)
        hab=request.form.get('habitaciones')

        # Nº de baños 1-10
        ba=request.form.get('baños')

        # Garaje (Si/No) Transformar a 1/0
        g=request.form.get('garaje')
        if g=='Si':
            gar=1
        else: 
            gar=0

        # Piscina (Si/No) Transformar a 1/0
        p=request.form.get('piscina')
        if p=='Si':
            pool=1
        else: 
            pool=0

        # Ascensor (Si/No) Transformar a 1/0
        e=request.form.get('ascensor')
        if e=='Si':
            elev=1
        else: 
            elev=0

        # Exterior (si/no) Transformar a 1/0
        ex=request.form.get('exterior')
        if ex=='Si':
            ext=1
        else: 
            ext=0

        # Datos originales que se envían a resultado.html para mostrar selección del cliente junto a la predicción
        # Añado m2 a Tamaño
        datos_originales_med=pd.DataFrame([[dist,t,str(area)+"m2",hab,ba,g,p,e,ex]]
                                      ,columns=['Distrito','Tipo','Tamaño', 'Habitaciones', 'Baños', 'Garaje', 'Piscina', 'Ascensor', 'Exterior'])

        
        # Valores para buscar clúster, escalado y predicción
        lista_valores_med=[[area,lat,lon,tipo,hab,ba,gar,pool,elev,ext]]

        # pasar datos a DF
        cliente_med=pd.DataFrame(lista_valores_med, columns=[['area','latitud','longitud','tipology','n_rooms','n_baths','has_garage','has_pool','has_elevator','is_exterior']])
        

        # escalar

        # se escala una copia para no perder datos originales
                
        escalado_med=cliente_med.copy()

        escalado_med=escalar_med(escalado_med, scaler_med)
        
        # buscar cluster
         
        clstr_med=int(func_kmeans_med.predict(escalado_med))

         # aplicar modelo

        if clstr_med==0 : 
            pred_med=cl0_med.predict(escalado_med)

        # si el cluster es 1,2 o 3 se aplica modelo 1
        elif ((clstr_med==1) | (clstr_med==2) | (clstr_med==3)) : 
            pred_med=cl1_med.predict(escalado_med)

        elif clstr_med==4 : 
            pred_med=cl4_med.predict(escalado_med)
        
        elif clstr_med==5 : 
            pred_med=cl5_med.predict(escalado_med)


        # Búsqueda de cercanos
        cerc_med=cercanos(area,lat, lon, clstr_med, hab, ba)

        # carga resultado.html y le pasa datos seleccionados por cliente y predicción
        return render_template('app_media_estancia/resultado_media.html', dat_orig=datos_originales_med, pred=int(pred_med), cercanos=cerc_med)

    except Exception as e:
        error_message = f"Error en la predicción: {str(e)}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
