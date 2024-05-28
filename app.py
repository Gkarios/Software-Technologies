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
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AffinityPropagation

tab1, tab2, tab3 = st.tabs(["Dimensionality Reduction", "Classification Algorithms", "Clustering Algorithms"])

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
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.title(f'{method} Visualization')
    
    st.pyplot(plt)

    st.write(f"{target_column_name} Histogram")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_column_name])
    st.pyplot(plt)

    st.write("Heatmap of Numerical Features")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

def calculate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mean_squared_error_val = mean_squared_error(y_true, y_pred)
    return accuracy, mean_squared_error_val

def calculate_clustering_metrics(X, clusters):
    silhouette_avg = silhouette_score(X, clusters)
    return silhouette_avg

with tab1:
    st.title('Dimensionality Reduction Techniques')

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        method = st.selectbox("Select Dimensionality Reduction Technique", ["PCA", "t-SNE"])

        target_column_name = df.columns[-1]
        
        perform_dimensionality_reduction(df, method, target_column_name)

with tab2:
    st.title("Classification Algorithms")

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
            n_neighbors = st.number_input("Select number of neighbors", min_value=1, max_value=20, value=5, step=1, key="n_neighbors")
            if n_neighbors:
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn_cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5)
                knn_model.fit(X_train, y_train)
                knn_y_pred = knn_model.predict(X_test)
                knn_accuracy = accuracy_score(y_test, knn_y_pred)
                st.write("Mean CV Score:", np.mean(knn_cv_scores))
                st.write("Accuracy on test set:", knn_accuracy)
                st.session_state.knn_accuracy = knn_accuracy

        with col2:
            st.subheader("Random Forest")
            n_estimators = st.number_input("Select number of trees", min_value=10, max_value=200, value=100, step=10, key="n_estimators")
            if n_estimators:
                rf_model = RandomForestClassifier(n_estimators=n_estimators)
                rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
                rf_model.fit(X_train, y_train)
                rf_y_pred = rf_model.predict(X_test)
                rf_accuracy = accuracy_score(y_test, rf_y_pred)
                st.write("Mean CV Score:", np.mean(rf_cv_scores))
                st.write("Accuracy on test set:", rf_accuracy)
                st.session_state.rf_accuracy = rf_accuracy
                
                if st.session_state.knn_accuracy and st.session_state.rf_accuracy:
                    if st.session_state.knn_accuracy > st.session_state.rf_accuracy:
                        st.markdown("**K-Nearest Neighbors has the better accuracy**")
                    elif st.session_state.rf_accuracy > st.session_state.knn_accuracy:
                        st.markdown("**Random Forest has the better accuracy**")
                    else:
                        st.markdown("**Both algorithms have the same accuracy**")
    else:
        st.write("Please upload a CSV file in the 'Dimensionality Reduction' tab.")

with tab3:
    st.title("Clustering Algorithms")

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

        st.subheader("K-Means Clustering")
        n_clusters_kmeans = st.number_input("Select number of clusters for K-Means", min_value=2, max_value=10, value=3, step=1, key="n_clusters_kmeans")
        if n_clusters_kmeans:
            kmeans_model = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
            cluster_labels_kmeans = kmeans_model.fit_predict(X_preprocessed)
            silhouette_avg_kmeans = calculate_clustering_metrics(X_preprocessed, cluster_labels_kmeans)
            st.write("K-Means Silhouette Score:", silhouette_avg_kmeans)

            st.write("K-Means Cluster Distribution:")
            st.write(pd.Series(cluster_labels_kmeans).value_counts())

            plt.figure(figsize=(10, 6))
            plt.scatter(X_preprocessed[:, 0], X_preprocessed[:, 1], c=cluster_labels_kmeans, cmap='viridis')
            plt.colorbar(label='Cluster')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('K-Means Clustering')
            st.pyplot(plt)

        st.subheader("Affinity Propagation Clustering")
        damping = 0.9
        preference = st.number_input("Select preference for Affinity Propagation", value=-50, step=1, key="preference_affinity")
        if preference:
            affinity_model = AffinityPropagation(damping=damping, preference=preference)
            cluster_labels_affinity = affinity_model.fit_predict(X_preprocessed)
            n_clusters_affinity = len(set(cluster_labels_affinity))
            st.write("Estimated number of clusters for Affinity Propagation:", n_clusters_affinity)

            silhouette_avg_affinity = calculate_clustering_metrics(X_preprocessed, cluster_labels_affinity)
            st.write("Affinity Propagation Silhouette Score:", silhouette_avg_affinity)

            st.write("Affinity Propagation Cluster Distribution:")
            st.write(pd.Series(cluster_labels_affinity).value_counts())

            plt.figure(figsize=(10, 6))
            plt.scatter(X_preprocessed[:, 0], X_preprocessed[:, 1], c=cluster_labels_affinity, cmap='viridis')
            plt.colorbar(label='Cluster')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('Affinity Propagation Clustering')
            st.pyplot(plt)
        else:
            st.write("Please select preference for Affinity Propagation.")

    else:
        st.write("Please upload a CSV file in the 'Dimensionality Reduction' tab.")
