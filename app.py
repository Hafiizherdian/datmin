import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from preprocessing import data_cleaning, handle_missing, encode_transform, feature_scaling

# Konfigurasi awal dashboard profesional
st.set_page_config(page_title='Dashboard Churn & Cluster', layout='wide')

st.markdown("""
<div style='padding: 1.5rem 0 1rem 0; border-bottom: 1px solid #eee; background: #fff;'>
    <h1 style='font-size:2.2rem; color:#222; margin-bottom:0;'>Telco Churn Prediction System</h1>
    <h3 style='font-size:1.3rem; color:#444; margin-top:0.5rem;'>Kelompok 12</h3>
    <b>Team Member</b>
    <table style='border-collapse:collapse; margin-top:0.5rem;'>
        <tr><th style='padding:4px 16px;border:1px solid #bbb;'>Nama</th><th style='padding:4px 16px;border:1px solid #bbb;'>NIM</th></tr>
        <tr><td style='padding:4px 16px;border:1px solid #bbb;'>Dara Saifa Mahiroh</td><td style='padding:4px 16px;border:1px solid #bbb;'>102022330396</td></tr>
        <tr><td style='padding:4px 16px;border:1px solid #bbb;'>Mariorie Queency Melanesia Moay</td><td style='padding:4px 16px;border:1px solid #bbb;'>102022300457</td></tr>
        <tr><td style='padding:4px 16px;border:1px solid #bbb;'>Hafiizh Herdian</td><td style='padding:4px 16px;border:1px solid #bbb;'>1202184194</td></tr>
    </table>
    <p style='font-size:1rem; color:#444; margin-top:1.2rem;'>Sistem ini menganalisis dataset pelanggan Telco untuk prediksi churn dan segmentasi otomatis berbasis machine learning.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image('https://static.streamlit.io/examples/dash-logo.png', width=80)
    st.title('Menu')
    uploaded_file = st.file_uploader('Pilih file CSV', type='csv')
    st.markdown('---')
    st.markdown('<span style="font-size:0.9rem;color:#888;">Dashboard by Kelompok 12</span>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Cek struktur kolom
        model_cols = joblib.load('scaler.joblib').feature_names_in_
        if not all([col in df.columns for col in model_cols]):
            st.error('Struktur kolom tidak sesuai dengan data training!\nKolom yang dibutuhkan: ' + ', '.join(model_cols))
        else:
            # Pipeline preprocessing
            df_clean = data_cleaning(df)
            df_clean = handle_missing(df_clean)
            df_clean = encode_transform(df_clean)
            X = df_clean[model_cols]
            scaler = joblib.load('scaler.joblib')
            X_scaled = scaler.transform(X)
            # Load model
            logreg = joblib.load('logreg_model.joblib')
            kmeans = joblib.load('kmeans_model.joblib')
            # Prediksi
            y_pred = logreg.predict(X_scaled)
            cluster = kmeans.predict(X_scaled)
            

            # Tab 1: Data Overview
            tab0, tab1, tab2, tab3, tab4 = st.tabs([
    'Dataset Overview',
    '1️⃣ Data Understanding', 
    '2️⃣ Preprocessing', 
    '3️⃣ Modeling & Evaluation', 
    '4️⃣ Export & Insight'])

            with tab0:
                st.markdown("""
    <div style='background:#f8fafc; border-radius:12px; padding:1.5rem 2rem 1rem 2rem; margin-bottom:1.5rem;'>
        <h2 style='color:#2563eb; margin-bottom:0.5rem;'>📊 Dataset Overview</h2>
        <span style='color:#555;'>Tahap awal: Memahami struktur, ringkasan data, dan hasil prediksi awal.</span>
    </div>
    """, unsafe_allow_html=True)
                # Card metrics
                met1, met2, met3 = st.columns(3)
                met1.metric('Jumlah Baris', df.shape[0])
                met2.metric('Jumlah Kolom', df.shape[1])
                met3.metric('Jumlah Fitur Numerik', len(df.select_dtypes(include=['float64', 'int64']).columns))
                st.markdown('---')
                # Section: Info Data
                with st.expander('🗂️ Struktur & Statistik Data', expanded=True):
                    st.write('**Tipe Data Tiap Kolom:**')
                    st.dataframe(pd.DataFrame(df.dtypes.astype(str), columns=['Tipe Data']))
                    st.write('**Statistik Deskriptif:**')
                    st.dataframe(df.describe(include='all').T)
                    st.write('**Preview Data:**')
                    st.dataframe(df.head(10), use_container_width=True)
                st.markdown('---')
                # Section: Distribusi Churn Asli
                if 'Churn' in df.columns:
                    st.markdown('#### <span style="color:#2563eb">Distribusi Data (Churn Asli jika ada)</span>', unsafe_allow_html=True)
                    fig_churn = px.histogram(df, x='Churn', color='Churn', barmode='group', text_auto=True,
                                            color_discrete_sequence=['#2563eb', '#22d3ee'])
                    fig_churn.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig_churn, use_container_width=True)
                st.markdown('---')
                # Hasil prediksi & cluster
                hasil = df.copy()
                hasil['Prediksi_Churn'] = y_pred
                hasil['Cluster'] = cluster
                st.success('Prediksi dan clustering selesai!')
                styled = hasil.head(20).style.applymap(lambda v: 'background-color:#fee2e2' if v==1 else ('background-color:#dcfce7' if v==0 else ''), subset=['Prediksi_Churn'])
                st.dataframe(styled, use_container_width=True)
                st.markdown('''<div style="margin-top:0.5rem;">
    <span style="background:#2563eb;color:#fff;padding:3px 8px;border-radius:5px;">Prediksi_Churn = 1</span> : Diprediksi <b>Churn</b> (akan berhenti)<br>
    <span style="background:#22d3ee;color:#222;padding:3px 8px;border-radius:5px;">Prediksi_Churn = 0</span> : Diprediksi <b>Tidak Churn</b><br>
    <span style="background:#bbf7d0;color:#222;padding:3px 8px;border-radius:5px;">Cluster 0 & 1</span> : Segmentasi otomatis K-Means (bukan label asli)
    </div>''', unsafe_allow_html=True)
                st.markdown('---')
                st.markdown('#### <span style="color:#2563eb">Distribusi Prediksi Churn (Logistic Regression)</span>', unsafe_allow_html=True)
                fig_pred = px.histogram(hasil, x='Prediksi_Churn', color='Prediksi_Churn', barmode='group', text_auto=True,
                           color_discrete_sequence=['#2563eb', '#22d3ee'])
                fig_pred.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig_pred, use_container_width=True, key='plotly_pred')
                st.markdown('#### <span style="color:#2563eb">Distribusi Cluster K-Means</span>', unsafe_allow_html=True)
                fig_clus = px.histogram(hasil, x='Cluster', color='Cluster', barmode='group', text_auto=True,
                           color_discrete_sequence=['#facc15', '#2563eb'])
                fig_clus.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig_clus, use_container_width=True, key='plotly_clus')

            with tab1:
                st.markdown("""
    <div style='background:#f0f9ff; border-radius:12px; padding:1.2rem 2rem 1rem 2rem; margin-bottom:1.5rem;'>
        <h3 style='color:#2563eb; margin-bottom:0.5rem;'>📊 1️⃣ Data Understanding</h3>
        <span style='color:#555;'>Memahami distribusi target dan deteksi outlier pada data pelanggan.</span>
    </div>
    """, unsafe_allow_html=True)
                st.markdown('---')
                with st.expander('🎯 Distribusi Target (Churn Asli jika ada)', expanded=True):
                    if 'Churn' in df.columns:
                        fig = px.histogram(df, x='Churn', color='Churn', barmode='group', text_auto=True,
                                           color_discrete_sequence=['#2563eb', '#22d3ee'])
                        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True, key=f'plotly_{hash(str(fig))}')
                    else:
                        st.info('Kolom Churn tidak tersedia di data.')
                st.markdown('---')
                with st.expander('🟦 Boxplot Outlier Fitur Numerik', expanded=True):
                    num_cols = hasil.select_dtypes(include=['float64', 'int64']).columns
                    if len(num_cols) > 0:
                        import plotly.graph_objects as go
                        fig_box = go.Figure()
                        for col in num_cols:
                            fig_box.add_trace(go.Box(y=hasil[col], name=col))
                        fig_box.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_box, use_container_width=True, key=f'plotly_box_{hash(str(fig_box))}')
                    else:
                        st.info('Tidak ada fitur numerik untuk boxplot.')

            with tab2:
                st.markdown("""
    <div style='background:#fefce8; border-radius:12px; padding:1.2rem 2rem 1rem 2rem; margin-bottom:1.5rem;'>
        <h3 style='color:#ca8a04; margin-bottom:0.5rem;'>🛠️ 2️⃣ Preprocessing</h3>
        <span style='color:#555;'>Pra-pemrosesan data: cleaning, imputasi, encoding, scaling.</span>
    </div>
    """, unsafe_allow_html=True)
                st.markdown('---')
                with st.expander('🔄 Ringkasan Preprocessing', expanded=True):
                    st.markdown('''
                    <ul>
                        <li>Data cleaning: menghapus duplikat, memperbaiki format</li>
                        <li>Imputasi missing value</li>
                        <li>Encoding kategori</li>
                        <li>Scaling numerik</li>
                    </ul>
                    ''', unsafe_allow_html=True)
                st.markdown('---')
                with st.expander('🧹 Preview Data Setelah Preprocessing', expanded=True):
                    st.dataframe(df_clean.head(10), use_container_width=True)
                st.markdown('---')
                with st.expander('📊 Distribusi Fitur Kategorikal (contoh)', expanded=False):
                    cat_cols = df_clean.select_dtypes(include=['uint8', 'int64']).columns
                    try:
                        for col in cat_cols:
                            if col != 'Churn' and df_clean[col].nunique() < 10:
                                fig = px.histogram(df_clean, x=col, title=f'Distribusi {col}', color_discrete_sequence=['#2563eb'])
                                st.plotly_chart(fig, use_container_width=True, key=f'plotly_{hash(str(fig))}')
                    except Exception as e:
                        st.info(f'Gagal menampilkan distribusi fitur kategorikal: {e}')
                with st.expander('📈 Distribusi Fitur Numerik (contoh)', expanded=False):
                    num_cols_clean = df_clean.select_dtypes(include=['float64', 'int64']).columns
                    try:
                        for col in num_cols_clean:
                            fig = px.histogram(df_clean, x=col, title=f'Distribusi {col}', color_discrete_sequence=['#22d3ee'])
                            st.plotly_chart(fig, use_container_width=True, key=f'plotly_{hash(str(fig))}')
                    except Exception as e:
                        st.info(f'Gagal menampilkan distribusi fitur numerik: {e}')

            with tab3:
                st.markdown("""
    <div style='background:#f1f5f9; border-radius:12px; padding:1.2rem 2rem 1rem 2rem; margin-bottom:1.5rem;'>
        <h3 style='color:#0e7490; margin-bottom:0.5rem;'>🤖 3️⃣ Modeling & Evaluation</h3>
        <span style='color:#555;'>Modeling, evaluasi prediksi, dan segmentasi pelanggan.</span>
    </div>
    """, unsafe_allow_html=True)
                st.markdown('---')
                met1, met2, met3 = st.columns(3)
                met1.metric('Jumlah Data', f"{df.shape[0]}")
                churn_rate = hasil['Prediksi_Churn'].mean() * 100
                met2.metric('Prediksi Churn (%)', f"{churn_rate:.1f}%")
                from sklearn.metrics import silhouette_score
                try:
                    sil_score = silhouette_score(X_scaled, cluster)
                    met3.metric('Silhouette Score', f"{sil_score:.2f}")
                except:
                    met3.metric('Silhouette Score', '-')
                st.markdown('---')
                with st.expander('📋 Tabel Hasil Prediksi & Cluster', expanded=True):
                    styled = hasil.head(30).style.applymap(lambda v: 'background-color:#fee2e2' if v==1 else ('background-color:#dcfce7' if v==0 else ''), subset=['Prediksi_Churn'])
                    st.dataframe(styled, use_container_width=True)
                    st.markdown('''<div style="margin-top:0.5rem;">
                    <span style="background:#2563eb;color:#fff;padding:3px 8px;border-radius:5px;">Prediksi_Churn = 1</span> : Diprediksi <b>Churn</b> (akan berhenti)<br>
                    <span style="background:#22d3ee;color:#222;padding:3px 8px;border-radius:5px;">Prediksi_Churn = 0</span> : Diprediksi <b>Tidak Churn</b><br>
                    <span style="background:#bbf7d0;color:#222;padding:3px 8px;border-radius:5px;">Cluster 0 & 1</span> : Segmentasi otomatis K-Means (bukan label asli)
                    </div>''', unsafe_allow_html=True)
                st.markdown('---')
                with st.expander('📊 Distribusi Prediksi & Cluster', expanded=True):
                    fig1 = px.histogram(hasil, x='Prediksi_Churn', color='Prediksi_Churn', barmode='group', text_auto=True,
                                       color_discrete_sequence=['#2563eb', '#22d3ee'])
                    fig1.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True, key='plotly_pred_churn')
                    fig2 = px.histogram(hasil, x='Cluster', color='Cluster', barmode='group', text_auto=True,
                                       color_discrete_sequence=['#facc15', '#2563eb'])
                    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True, key='plotly_cluster')
                st.markdown('---')
                with st.expander('📑 Karakteristik & Evaluasi Cluster', expanded=False):
                    cluster_summary = hasil.groupby('Cluster')[num_cols].mean().round(2)
                    st.write('Rata-rata fitur numerik tiap cluster:')
                    st.dataframe(cluster_summary)
                    st.write('Jumlah anggota tiap cluster:')
                    st.write(hasil['Cluster'].value_counts())
                    if 'Churn' in hasil.columns:
                        st.write('Distribusi Churn pada tiap cluster:')
                        churn_per_cluster = hasil.groupby('Cluster')['Churn'].value_counts().unstack().fillna(0)
                        st.dataframe(churn_per_cluster)
                        fig3 = px.bar(churn_per_cluster, barmode='group', title='Distribusi Churn pada Tiap Cluster')
                        st.plotly_chart(fig3, use_container_width=True, key='plotly_churn_cluster')
                st.markdown('---')
                with st.expander('⭐ Feature Importance (Koefisien Model)', expanded=False):
                    feature_importance = logreg.coef_[0]
                    feature_names = model_cols
                    feature_importance_df = pd.DataFrame({'Fitur': feature_names, 'Importance': feature_importance})
                    feature_importance_df = feature_importance_df.reindex(feature_importance_df['Importance'].abs().sort_values(ascending=False).index)
                    fig_fi = px.bar(feature_importance_df.head(10), x='Importance', y='Fitur', orientation='h', title='Feature Importance (Top 10)')
                    st.plotly_chart(fig_fi, use_container_width=True, key='plotly_feat_imp')
                    st.write(feature_importance_df.head(10))
                st.markdown('---')
                with st.expander('🔁 Cross-validation Logistic Regression', expanded=False):
                    from sklearn.model_selection import cross_val_score
                    X = df_clean[model_cols]
                    y = df_clean['Churn'] if 'Churn' in df_clean.columns else None
                    if y is not None:
                        logreg = joblib.load('logreg_model.joblib')
                        scores = cross_val_score(logreg, X_scaled, y, cv=5)
                        st.write(f'Skor cross-validation (5-fold): {scores}')
                        st.write(f'Rata-rata: {scores.mean():.2f}, Std: {scores.std():.2f}')
                    else:
                        st.info('Kolom Churn tidak ada, cross-validation tidak bisa dilakukan.')
                st.markdown('---')
                with st.expander('🟦 Akurasi & Confusion Matrix', expanded=False):
                    if 'Churn' in df.columns:
                        from sklearn.metrics import accuracy_score, confusion_matrix
                        acc = accuracy_score(df_clean['Churn'], y_pred)
                        st.info(f'Akurasi model pada data ini: {acc:.2f}')
                        st.write('Confusion Matrix:')
                        import plotly.figure_factory as ff
                        cm = confusion_matrix(df_clean['Churn'], y_pred)
                        z = cm
                        x = ['Prediksi Tidak Churn', 'Prediksi Churn']
                        y_ = ['Asli Tidak Churn', 'Asli Churn']
                        fig_cm = ff.create_annotated_heatmap(z, x=x, y=y_, colorscale='Blues')
                        st.plotly_chart(fig_cm, use_container_width=True, key='plotly_confmat')
                st.markdown('---')
                with st.expander('📏 Silhouette Score K-Means', expanded=False):
                    try:
                        sil_score = silhouette_score(X_scaled, cluster)
                        st.write(f'Silhouette Score: {sil_score:.2f} (semakin mendekati 1, cluster makin baik)')
                    except Exception as e:
                        st.info(f'Gagal menghitung silhouette score: {e}')
            if y is not None:
                logreg = joblib.load('logreg_model.joblib')
                scores = cross_val_score(logreg, X_scaled, y, cv=5)
                st.write(f'Skor cross-validation (5-fold): {scores}')
                st.write(f'Rata-rata: {scores.mean():.2f}, Std: {scores.std():.2f}')
            else:
                st.info('Kolom Churn tidak ada, cross-validation tidak bisa dilakukan.')

            # Silhouette score K-Means
            st.subheader('Silhouette Score K-Means')
            from sklearn.metrics import silhouette_score
            try:
                sil_score = silhouette_score(X_scaled, cluster)
                st.write(f'Silhouette Score: {sil_score:.2f} (semakin mendekati 1, cluster makin baik)')
            except Exception as e:
                st.info(f'Gagal menghitung silhouette score: {e}')

            # Akurasi jika label Churn ada
            if 'Churn' in df.columns:
                from sklearn.metrics import accuracy_score, confusion_matrix
                acc = accuracy_score(df_clean['Churn'], y_pred)
                st.info(f'Akurasi model pada data ini: {acc:.2f}')
                st.write('Confusion Matrix:')
                st.write(pd.DataFrame(confusion_matrix(df_clean['Churn'], y_pred),
                                     columns=['Prediksi Tidak Churn', 'Prediksi Churn'],
                                     index=['Asli Tidak Churn', 'Asli Churn']))

            with tab4:
                st.header('📤 4️⃣ Export & Insight')
                st.markdown('Tahap akhir data mining: Ekspor hasil prediksi & insight bisnis.')
                st.download_button('Download hasil (CSV)', data=hasil.to_csv(index=False), file_name='hasil_prediksi.csv', mime='text/csv', key='download_prediksi_tab4')
                st.markdown('**Catatan & Insight Bisnis:**')
                st.info('Gunakan hasil prediksi churn dan segmentasi cluster untuk strategi retensi pelanggan dan penawaran personalisasi.')

    except Exception as e:
        st.error(f'Gagal memproses file: {e}')
    else:
        st.info('Silakan upload file untuk mulai prediksi dan clustering.')
