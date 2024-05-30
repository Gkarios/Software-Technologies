import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AffinityPropagation
import io

tab1, tab2, tab3 = st.tabs(["2D Οπτικοποίηση", "Αλγόριθμοι κατηγοριοποίησης", "Αλγόριθμοι ομαδοποίησης"])

def perform_dimensionality_reduction(df, method, target_column_name):
    numerical_features = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    if method == "PCA":
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('dim_reduction', PCA(n_components=2))
        ])
    elif method == "t-SNE":
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('dim_reduction', TSNE(n_components=2, random_state=42))
        ])

    X_reduced = pipeline.fit_transform(df)

    target_column = None
    if target_column_name in df.columns:
        target_column = df[target_column_name]
        if target_column.dtype == 'object':
            label_encoder = LabelEncoder()
            target_column = label_encoder.fit_transform(target_column)

    plt.figure(figsize=(10, 6))
    if target_column is not None:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=target_column, cmap='plasma')
        plt.colorbar(label='Target')
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    plt.xlabel(f'{method} Principal Component 1')
    plt.ylabel(f'{method} Principal Component 2')
    plt.title(f'Οπτικοποίηση {method} ')

    st.pyplot(plt)

    st.write(f" Ιστόγραμμα {target_column_name}")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_column_name])
    st.pyplot(plt)

    st.write("Heatmap των χαρακτηριστικών με συνεχείς τιμές")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mean_squared_error_val = mean_squared_error(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, mean_squared_error_val, conf_matrix

def calculate_clustering_metrics(X, clusters):
    silhouette_avg = silhouette_score(X, clusters)
    return silhouette_avg

with tab1:
    st.title('2D Οπτικοποιήσεις')

    uploaded_file = st.file_uploader("Ανεβάστε csv ή excel αρχείο", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        method = st.selectbox("Επιλέξτε μέθοδο 2D οπτικοποίησης", ["PCA", "t-SNE"])

        target_column_name = df.columns[-1]

        perform_dimensionality_reduction(df, method, target_column_name)

with tab2:
    st.title("Αλγόριθμοι κατηγοριοποίησης")

    if uploaded_file is not None:
        target_column_name = df.columns[-1]

        numerical_features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        if target_column_name in numerical_features:
            numerical_features.remove(target_column_name)
        if target_column_name in categorical_features:
            categorical_features.remove(target_column_name)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        X = df.drop(columns=[target_column_name])
        y = df[target_column_name]

        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        X = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("K-Nearest Neighbors")
            n_neighbors = st.number_input("Επιλέξτε πλήθος γειτόνων", min_value=1, max_value=20, value=5, step=1, key="n_neighbors")
            if n_neighbors:
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5)
                knn_model.fit(X_train, y_train)
                knn_y_pred = knn_model.predict(X_test)
                knn_accuracy, knn_mse, knn_conf_matrix = calculate_classification_metrics(y_test, knn_y_pred)
                st.write("Μέση απόδοση cross validation:", np.mean(knn_cv_scores))
                st.write("Ακρίβεια αλγορίθμου:", knn_accuracy)
                st.write("Πίνακας σύγχυσης:")
                st.write(knn_conf_matrix)
                st.session_state.knn_accuracy = knn_accuracy

        with col2:
            st.subheader("Random Forest")
            n_estimators = st.number_input("Επιλέξτε πλήθος δέντρων ", min_value=10, max_value=200, value=100, step=10, key="n_estimators")
            if n_estimators:
                rf_model = RandomForestClassifier(n_estimators=n_estimators)
                rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
                rf_model.fit(X_train, y_train)
                rf_y_pred = rf_model.predict(X_test)
                rf_accuracy, rf_mse, rf_conf_matrix = calculate_classification_metrics(y_test, rf_y_pred)
                st.write("Μέση απόδοση cross validation:", np.mean(rf_cv_scores))
                st.write("Ακρίβεια αλγορίθμου:", rf_accuracy)
                st.write("Πίνακας σύγχυσης:")
                st.write(rf_conf_matrix)
                st.session_state.rf_accuracy = rf_accuracy

                if 'knn_accuracy' in st.session_state and 'rf_accuracy' in st.session_state:
                    if st.session_state.knn_accuracy > st.session_state.rf_accuracy:
                        st.markdown("** Ο αλγόριθμος Κ-Κοντινότεροι γείτονες έχει μεγαλύτερη ακρίβεια**")
                    elif st.session_state.rf_accuracy > st.session_state.knn_accuracy:
                        st.markdown("** Ο αλγόριθμος Τυχαίο δάσος έχει μεγαλύτερη ακρίβεια**")
                    else:
                        st.markdown("**Οι 2 αλγόριθμοι έχουν την ίδια ακρίβεια**")
    else:
        st.write("Παρακαλώ ανεβάστε αρχείο στην σελίδα της 2D οπτικοποίησης")

with tab3:
    st.title("Αλγόριθμοι ομαδοποίησης")

    if uploaded_file is not None:

        target_column_name = df.columns[-1]

        numerical_features = df.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        if target_column_name in numerical_features:
            numerical_features.remove(target_column_name)
        if target_column_name in categorical_features:
            categorical_features.remove(target_column_name)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        X = df.drop(columns=[target_column_name])
        X_preprocessed = preprocessor.fit_transform(X)

        st.subheader("Oμαδοποίηση K-Means ")
        n_clusters_kmeans = st.number_input("Επιλέξτε αριμθό clusters(Διαφορετικών ομάδων)", min_value=2, max_value=10, value=3, step=1, key="n_clusters_kmeans")
        if n_clusters_kmeans:
            kmeans_model = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
            cluster_labels_kmeans = kmeans_model.fit_predict(X_preprocessed)
            silhouette_avg_kmeans = calculate_clustering_metrics(X_preprocessed, cluster_labels_kmeans)
            st.write("K-Means Silhouette Score:", silhouette_avg_kmeans)

            st.write("K-Means κατανομή των clusters:")
            st.write(pd.Series(cluster_labels_kmeans).value_counts())

            plt.figure(figsize=(10, 6))
            plt.scatter(X_preprocessed[:, 0], X_preprocessed[:, 1], c=cluster_labels_kmeans, cmap='viridis')
            plt.colorbar(label='Cluster')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Ομαδοποίηση K-Means')
            st.pyplot(plt)
            st.session_state.kmeans_silhouette_avg = silhouette_avg_kmeans

        st.subheader("Affinity propagation(Διάδοση συνάφειας) ")
        damping = 0.9
        preference = st.number_input("Επιλέξτε μέτρο προτίμησης", value=-50, step=1, key="preference_affinity")
        if preference:
            affinity_model = AffinityPropagation(damping=damping, preference=preference)
            cluster_labels_affinity = affinity_model.fit_predict(X_preprocessed)
            n_clusters_affinity = len(set(cluster_labels_affinity))
            st.write("Αριθμός clusters για διάδοση συνάφειας:", n_clusters_affinity)

            silhouette_avg_affinity = calculate_clustering_metrics(X_preprocessed, cluster_labels_affinity)
            st.write("Silhouette Score:", silhouette_avg_affinity)

            st.write(" Κατανομή clusters για διάδοση συνάφειας:")
            st.write(pd.Series(cluster_labels_affinity).value_counts())

            plt.figure(figsize=(10, 6))
            plt.scatter(X_preprocessed[:, 0], X_preprocessed[:, 1], c=cluster_labels_affinity, cmap='viridis')
            plt.colorbar(label='Cluster')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Ομαδοποίηση διάδοσης συνάφειας')
            st.pyplot(plt)
            st.session_state.affinity_silhouette_avg = silhouette_avg_affinity

        if 'kmeans_silhouette_avg' in st.session_state and 'affinity_silhouette_avg' in st.session_state:
            if st.session_state.kmeans_silhouette_avg > st.session_state.affinity_silhouette_avg:
                st.markdown("** Ο αλγόριθμος K-Means έχει μεγαλύτερο Silhouette Score**")
            elif st.session_state.affinity_silhouette_avg > st.session_state.kmeans_silhouette_avg:
                st.markdown("** Ο αλγόριθμος Διάδοσης συνάφειας έχει μεγαλύτερο Silhouette Score**")
            else:
                st.markdown("**Οι 2 αλγόριθμοι έχουν το ίδιο Silhouette Score**")

    else:
        st.write("Παρακαλώ ανεβάστε αρχείο στην σελίδα της 2D οπτικοποίησης.")
