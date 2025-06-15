# Librerías necesarias
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ============================
# Configuración de la página
# ============================

st.set_page_config(page_title="Modelo de Mantenimiento Preventivo", layout="wide")

st.markdown(
    """
    <style>
        .main {background-color: #e4dadf;}
        h1, h2, h3 {color: #0c1013;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
#  Login setup
# ======================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Login
if not st.session_state.logged_in:
    placeholder = st.empty()

    with placeholder.form("login"):
        # Configuración del cover logo
        st.image("images/arcacontinental_cover.jpeg", use_container_width=True, width=200)
        st.markdown("<h1 style='text-align: center;'>¡Bienvenido!</h1>", unsafe_allow_html=True)
        st.markdown("### Iniciar sesión")
        username = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        submit = st.form_submit_button("Iniciar sesión")

    if submit:
        if (username == "admin" and password == "admin") or (username == "user" and password == "user"):
            st.session_state.logged_in = True
            st.session_state.username = username
            placeholder.empty()
            st.success(f"Inicio de sesión exitoso para {username.upper()}")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")

if (st.session_state.logged_in == True):
    
    # ======================
    # Logo y título
    # ======================

    st.image("images/arcacontinental_cover.jpeg", width=1500) 
    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #bb0a01 0%, #ffc914 100%); padding: 1.5rem 0; border-radius: 18px; margin-bottom: 1.5rem;'>
            <h1 style='text-align: center; color: #fff; font-family: Montserrat, sans-serif; font-weight: bold; margin-top: 1rem;'>
                Modelo Predictivo de Fallas en Coolers
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================
    # Carga de modelos y datos
    # ============================

    # Modelo predictor
    model = joblib.load("predictor_de_fallas.pkl")

    #Carga de datos
    @st.cache_data
    def load_data(uploaded_file):
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        return None
    
    st.markdown(
        """
        <h2 style='color: #701e19; font-family: Montserrat, sans-serif; font-weight: bold; margin-bottom: 1rem; text-align: center;'>
            ¡Bienvenido!
        </h2>
        <p style='color: #333; font-size: 1.2rem; font-family: Montserrat, sans-serif; text-align: center;'>
            Tu app de confianza está lista para ayudarte a predecir las fallas de tus coolers<br>
            y proporcionarte insights valiosos para tu operación.
        </p>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        " Carga un archivo CSV con tus datos de consulta:",
        type=["csv"],
        accept_multiple_files=False,
        help="El archivo debe estar en formato CSV.",
        label_visibility="visible"
    )
    
    df = load_data(uploaded_file)
    if uploaded_file is not None:
        if df is not None:
            st.success("¡Datos cargados exitosamente!")
        else:
            st.error("Error al cargar los datos. Por favor, verifica el archivo CSV.")
            
    # ============================
    # Predicciones
    # ============================
            
    if df is not None:
        if st.button("Predecir", use_container_width=True):
            # Predicciones usando el modelo desarrollado
            try:
                # ============================
                # Preprocesamiento
                # ============================
                
                # Crear features más ricos por cooler
                numeric_cols = df.select_dtypes(include='number').columns.drop(['cooler_id', 'calday', 'calmonth'], errors='ignore')
                coolers_agg = df.groupby('cooler_id')[numeric_cols].agg(['mean', 'min', 'max'])
                coolers_agg.columns = ['_'.join(col).strip() for col in coolers_agg.columns.values]
                coolers_agg.reset_index(inplace=True)

                # Calcular deltas (último - primero) por variable numérica
                delta_df = pd.DataFrame()
                for col in df.select_dtypes(include='number').columns:
                    if col not in ['cooler_id', 'calday', 'calmonth']:
                        first = df.groupby('cooler_id')[col].first()
                        last = df.groupby('cooler_id')[col].last()
                        delta_df[f'{col}_delta'] = last - first

                # Unir los deltas a coolers_agg
                delta_df = delta_df.reset_index()
                coolers_agg = coolers_agg.merge(delta_df, on='cooler_id', how='left')

                # Variable de longitud (cuántos registros tiene cada cooler)
                coolers_agg['n_observations'] = df.groupby('cooler_id').size().values
                
                # Intentar eliminar 'calday' y 'calmonth' si existen, si no, continuar
                cols_to_drop = [col for col in ['cooler_id', 'calday', 'calmonth'] if col in coolers_agg.columns]
                X = coolers_agg.drop(columns=cols_to_drop)
                
                # Escalado
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                probs = np.round(model.predict_proba(X_scaled)[:, 1], 4)
                # Forzar a que siempre tenga 4 decimales como string
                probs_str = [f"{p:.4f}" for p in probs]
                output_df = pd.DataFrame({
                    'cooler_id': coolers_agg['cooler_id'],
                    'failure_probability': probs_str
                })
                st.markdown(
                    """
                    <div style='background: linear-gradient(90deg, #bb0a01 0%, #ffc914 100%); padding: 1.5rem 0; border-radius: 18px; margin-bottom: 1.5rem;'>
                        <h4 style='text-align: center; color: #fff; font-family: Montserrat, sans-serif; font-weight: bold; margin-top: 1rem; letter-spacing: 1px; font-size: 2rem;'>
                            Predicciones de Fallas por Cooler <br>
                            <span style='font-size: 1.2rem; font-weight: 500; color: #ffe082; font-family: Montserrat, sans-serif;'>
                                Consulta los resultados de predicción para tus equipos.<br>
                                ¡Prioriza la atención según el riesgo!
                            </span>
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # DataFrame estilizado sin la columna 'Riesgo'
                styled_df = output_df[['cooler_id', 'failure_probability']].style.background_gradient(
                    subset=['failure_probability'],
                    cmap='Reds'
                )

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "cooler_id": "ID Cooler",
                        "failure_probability": st.column_config.Column(
                            "Probabilidad de Falla",
                            help="Probabilidad estimada de falla (0-1)",
                        )
                    }
                )
            
            except Exception as e:
                st.error(f"Ocurrió un error al predecir: {e}")
                
            # ============================
            # Insights de predicciones
            # ============================
                
            # Clasificación de riesgo según la probabilidad de falla
            def clasificar_riesgo(prob):
                prob = float(prob) * 100  # convertir a porcentaje
                if prob >= 90:
                    return "Urgente"
                elif prob >= 60:
                    return "Alto"
                elif prob >= 30:
                    return "Medio"
                else:
                    return "Bajo"

            output_df['Riesgo'] = output_df['failure_probability'].apply(clasificar_riesgo)
                
            # Métricas de riesgo
            riesgo_cats = output_df['Riesgo'].value_counts().reindex(
                    ["Urgente", "Alto", "Medio", "Bajo"], fill_value=0)
            
            st.markdown(
                """
                <div style='background: linear-gradient(90deg, #bb0a01 0%, #ffc914 100%); padding: 1.5rem 0; border-radius: 18px; margin-bottom: 1.5rem;'>
                        <h4 style='text-align: center; color: #fff; font-family: Montserrat, sans-serif; font-weight: bold; margin-top: 1rem; letter-spacing: 1px; font-size: 2rem;'>
                            Insights de las Predicciones Realizadas <br>
                            <span style='font-size: 1.2rem; font-weight: 500; color: #ffe082; font-family: Montserrat, sans-serif;'>
                                Visualiza el resumen de riesgos y la distribución de probabilidades para tus coolers. <br>
                            </span>
                        </h4>
                    </div>
                """,
                unsafe_allow_html=True
            )

            total = len(output_df)
            # Una métricas por categoría de riesgo
            metric_cols = st.columns(4, gap="large")
            riesgo_labels = ["Urgente", "Alto", "Medio", "Bajo"]
            metric_colors = ["#bb0a01", "#e4572e", "#ffc914", "#4caf50"]
            for i, (cat, color) in enumerate(zip(riesgo_labels, metric_colors)):
                count = riesgo_cats[cat]
                percent = (count / total) * 100 if total > 0 else 0
                with metric_cols[i]:
                    st.markdown(
                        f"<div style='display: flex; flex-direction: column; align-items: center;'>"
                        f"<span style='font-size: 1.1rem; color: {color}; font-weight: bold;'>{cat}</span>"
                        f"<span style='font-size: 2.2rem; color: {color}; font-weight: bold;'>{count}</span>"
                        f"<span style='font-size: 1.1rem; color: #333;'>{percent:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Gráfica de pie debajo de las métricas
            fig = px.pie(
                names=riesgo_cats.index,
                values=riesgo_cats.values,
                color=riesgo_cats.index,
                color_discrete_map={
                    "Urgente": "#bb0a01",
                    "Alto": "#e4572e",
                    "Medio": "#ffc914",
                    "Bajo": "#4caf50"
                },
                title=""
            )
            fig.update_traces(textinfo='percent+label', textfont_size=14, pull=[0.08, 0.06, 0, 0])
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=13, family='Montserrat, sans-serif')
                ),
                margin=dict(t=10, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Ahorros por predicciones
            # Merge con output_df usando cooler_id como clave
            output_df = output_df.merge(
            coolers_agg[['cooler_id', 'amount_mean']].rename(columns={'amount_mean': 'amount_per_cooler'}),
            on='cooler_id',
            how='inner'  # Mantenemos solo los coolers que existen en ambos
            )
            # Cálculo de ahorros potenciales por riesgo
            ahorro_urgente = output_df.loc[output_df['Riesgo'] == 'Urgente', 'amount_per_cooler'].sum()
            ahorro_alto = output_df.loc[output_df['Riesgo'] == 'Alto', 'amount_per_cooler'].sum()

            # Mostrar métricas de ahorro de forma llamativa
            st.markdown(
                """
                <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 2rem; margin-bottom: 2rem;'>
                    <div style='background: #bb0a01; color: #fff; padding: 1.5rem 2.5rem; border-radius: 18px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <span style='font-size: 1.1rem; font-weight: 500;'>Ahorro Potencial (Urgente)</span><br>
                        <span style='font-size: 2.2rem; font-weight: bold;'>${:,.2f}</span>
                    </div>
                    <div style='background: #e4572e; color: #fff; padding: 1.5rem 2.5rem; border-radius: 18px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <span style='font-size: 1.1rem; font-weight: 500;'>Ahorro Potencial (Alto)</span><br>
                        <span style='font-size: 2.2rem; font-weight: bold;'>${:,.2f}</span>
                    </div>
                </div>
                """.format(ahorro_urgente, ahorro_alto),
                unsafe_allow_html=True
            )

            # Área de recomendaciones
            st.markdown(
                """
                <div style='background: linear-gradient(90deg, #bb0a01 0%, #ffc914 100%); padding: 1.5rem 2rem; border-radius: 18px; margin-top: 1.5rem;'>
                    <h3 style='color: #fff; font-family: Montserrat, sans-serif; font-weight: bold; margin-bottom: 0.5rem;'>
                        Recomendaciones
                    </h3>
                    <ul style='color: #fff; font-size: 1.1rem; font-family: Montserrat, sans-serif;'>
                        <li><b>Prioriza la atención</b> a los coolers clasificados como <span style='color:#ffe082;'>Urgente</span> y <span style='color:#ffe082;'>Alto</span> para maximizar el ahorro y reducir el riesgo de fallas críticas.</li>
                        <li>Programa mantenimientos preventivos en estos equipos lo antes posible.</li>
                        <li>Monitorea periódicamente los coolers con riesgo <b>Medio</b> y <b>Bajo</b> para anticipar posibles cambios en su condición.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            
            
            

                
                
    