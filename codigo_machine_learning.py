import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.calibration import CalibratedClassifierCV

def find_best_threshold_f1(y_true, y_prob, n_thresholds=200):
    thresholds = np.linspace(0, 1, n_thresholds)
    
    best_threshold = 0.5
    best_f1 = -1
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    return best_threshold, best_f1

if __name__ == "__main__":

    # =========================
    # 1. Carregar dados
    # =========================
    df = pd.read_csv("amostra_estratificada.csv", sep=";")

    # =========================
    # 2. Definir variáveis
    # =========================
    features_num = [
        "valor_financiado",
        "taxa_juros",
        "tempo_relacionamento",
        "escore_fico",
        "inadimplencia_2_anos",
        "registros_publicos_negativos",
        "renda_mensal"
    ]

    features_cat = ["finalidade_emprestimo"]

    target = "default"

    X = df[features_num + features_cat].copy()
    y = df[target].copy()

    # Garantir que a variável alvo seja numérica binária
    # Se já estiver em 0 e 1, esta linha não altera nada
    y = y.astype(int)

    # =========================
    # 3. Pré-processamento
    # =========================
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), features_num), ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat)])

    # =========================
    # 4. Definir modelos
    # =========================
    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": CalibratedClassifierCV(estimator=LinearSVC(random_state=42, max_iter=5000),cv=3),
        "MLP": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42),
        "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }

    # =========================
    # 5. Validação cruzada
    # =========================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #cv, _ = train_test_split(df, test_size=0.20, stratify=df["default"], random_state=42)

    results = []

    plt.figure(figsize=(10, 8))

    for model_name, model in models.items():
        # Pipeline completo: pré-processamento + modelo
        print(model_name)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Probabilidades previstas por validação cruzada
        y_prob = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        # Encontrar melhor threshold
        best_t, best_f1 = find_best_threshold_f1(y, y_prob)

        # Gerar predições com threshold otimizado
        y_pred = (y_prob >= best_t).astype(int)

        # Métricas
        limiar = best_t
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = best_f1
        auc = roc_auc_score(y, y_prob)

        # KS
        df_ks = pd.DataFrame({"y_true": y, "y_prob": y_prob})
        df_ks = df_ks.sort_values("y_prob", ascending=True).reset_index(drop=True)

        total_bad = (df_ks["y_true"] == 1).sum()
        total_good = (df_ks["y_true"] == 0).sum()

        df_ks["cum_bad"] = (df_ks["y_true"] == 1).cumsum() / total_bad
        df_ks["cum_good"] = (df_ks["y_true"] == 0).cumsum() / total_good
        ks = np.max(np.abs(df_ks["cum_bad"] - df_ks["cum_good"]))

        results.append({
            "Modelo": model_name,
            "Lmiar": limiar,
            "Acurácia": acc,
            "Precisão": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": auc,
            "KS": ks
        })

        # Curva ROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

        cm = confusion_matrix(y, y_pred)

        # ==========================================
        # Plot
        # ==========================================
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

        plt.title(f"Matriz de Confusão – {model_name}")
        plt.xlabel("Previsto")
        plt.ylabel("Real")

        plt.xticks([0.5, 1.5], ["Adimplente", "Inadimplente"])
        plt.yticks([0.5, 1.5], ["Adimplente", "Inadimplente"], rotation=0)

        plt.tight_layout()
        plt.savefig(f"./figuras/{model_name}_mat_conf.png", dpi=300, bbox_inches="tight")
        #plt.show()

    # =========================
    # 6. DataFrame de resultados
    # =========================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)
    results_df.to_csv("resultados_modelos.csv", sep=";", index=False)
    print(results_df)

    # =========================
    # 7. Plot da curva ROC
    # =========================
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        y_prob = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC dos Modelos de Machine Learning")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figuras/curva_roc_modelos.png", dpi=300, bbox_inches="tight")
    plt.show()