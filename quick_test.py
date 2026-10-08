import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("--- FluvialDrainage-ML Quick Test ---")
print("\n--- 1. Clustering Analysis ---")

use_feature_engineering = True
features_base = ["DIST_DN_KM", "DIS_AV_CMS", "ORD_CLAS", "ORD_FLOW", "ORD_STRA", "DIST_UP_KM", "UPLAND_SKM", "CATCH_SKM"]
features = ["DIST_DN_KM", "NET_LENGTH", "ORDER_RATIO", "LENGTH_OVER_ORDER", "UPLAND_DENSITY"]

def engineer_features(df):
    df = df.copy()
    epsilon = 1e-5
    df['NET_LENGTH'] = df['DIST_UP_KM'] + df['DIST_DN_KM']
    df['ORDER_RATIO'] = df['ORD_FLOW'] / (df['ORD_CLAS'] + epsilon)
    df['LENGTH_OVER_ORDER'] = df['NET_LENGTH'] / (df['ORD_FLOW'] + epsilon)
    df['UPLAND_DENSITY'] = df['UPLAND_SKM'] / (df['DIS_AV_CMS'] + epsilon)
    return df

# Load rivers
rivers_df = pd.read_csv("data/rivers.csv", sep=';')
rivers_features = rivers_df[["HYRIV_ID"] + features_base].copy()
if use_feature_engineering:
    rivers_features = engineer_features(rivers_features)

data_array_raw = rivers_features[features].values
ids_raw = rivers_features["HYRIV_ID"].values 

# Preprocessing for clustering
clust_imputer = SimpleImputer(strategy="mean")
data_array_imputed = clust_imputer.fit_transform(data_array_raw)
clust_scaler = PowerTransformer()
clean_scaled_data = clust_scaler.fit_transform(data_array_imputed)

k = 7
print("Running K-Means (k=7) on the full dataset...")
kmeans = KMeans(n_clusters=k, init='random', n_init=13, random_state=42)
kmeans_labels = kmeans.fit_predict(clean_scaled_data)

print(f"Successfully assigned {len(kmeans_labels)} river segments to 7 clusters.")

print("\n[Clustering Metrics & Label Mapping]")
labels_folder = "data/Merged_labels"
labelled_dfs = []
for file in glob.glob(f"{labels_folder}/*.csv"):
    if file.lower().endswith("rivers.csv") or file.lower().endswith("new.csv"): continue
    class_name = os.path.splitext(os.path.basename(file))[0].capitalize()
    df = pd.read_csv(file, sep=';')
    df = df[["HYRIV_ID"]].copy()
    df["True_Label"] = class_name
    labelled_dfs.append(df)

if labelled_dfs:
    true_labels_df = pd.concat(labelled_dfs, ignore_index=True)
    results_df = pd.DataFrame({"HYRIV_ID": ids_raw, "KMeans_Cluster": kmeans_labels})
    mapped_df = pd.merge(results_df, true_labels_df, on="HYRIV_ID", how="inner")
    unique_classes = sorted(true_labels_df["True_Label"].unique())
    n_classes = len(unique_classes)
    
    cost_matrix = np.zeros((k, n_classes))
    for i in range(k):
        for j, cls_name in enumerate(unique_classes):
            count = len(mapped_df[(mapped_df["KMeans_Cluster"] == i) & (mapped_df["True_Label"] == cls_name)])
            cost_matrix[i, j] = -count 
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    kmeans_cluster_names = {r: unique_classes[c] for r, c in zip(row_ind, col_ind)}
    
    print("Cluster -> Geological Class (Majority Voting):")
    for cl_id, cl_name in kmeans_cluster_names.items():
        print(f"  Cluster {cl_id}: {cl_name}")
else:
    kmeans_cluster_names = {i: f"Cluster {i}" for i in range(k)}

print("\n--- 2. Supervised Classification ---")
# Feature engineering for classification
labelled_dfs_class = []
label_map = {}
for file in glob.glob(f"{labels_folder}/*.csv"):
    if file.lower().endswith("rivers.csv") or file.lower().endswith("new.csv"): continue
    class_name = os.path.splitext(os.path.basename(file))[0].capitalize()
    df = pd.read_csv(file, sep=';')
    df = df[["HYRIV_ID"]]
    df["label"] = class_name
    labelled_dfs_class.append(df)
    label_map[class_name] = file

labels_df = pd.concat(labelled_dfs_class, ignore_index=True)
labelled_df = pd.merge(rivers_features, labels_df, on="HYRIV_ID", how="inner")
print(f"Total labelled rivers: {len(labelled_df)}")

X_labelled_raw = labelled_df[features].values
y_labelled = labelled_df["label"].values

le = LabelEncoder()
y_labelled_encoded = le.fit_transform(y_labelled)
id_labelled = labelled_df["HYRIV_ID"].values

X_train_raw, X_test_raw, y_train, y_test, id_train, id_test = train_test_split(
    X_labelled_raw, y_labelled_encoded, id_labelled,
    test_size=0.2, random_state=42, stratify=y_labelled_encoded
)

class_imputer = SimpleImputer(strategy="mean")
X_train_imputed = class_imputer.fit_transform(X_train_raw)
class_scaler = StandardScaler()
X_train_scaled = class_scaler.fit_transform(X_train_imputed)

y_encoded_all = le.transform(y_labelled)
classes_encoded = np.unique(y_encoded_all)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_encoded, y=y_encoded_all)
class_weight_dict = dict(zip(classes_encoded, class_weights))

print("Training pseudo-labeled models for Ensemble (RandomForest, XGBoost, LightGBM, HistGradientBoosting)...")

test_hyriv_ids = id_test.tolist()

def train_models(model_type, num_iterations, class_thresholds=None, min_new_labels=500, max_per_class=None):
    labelled_local = labelled_df.copy()
    X_labelled_imp = class_imputer.transform(labelled_local[features])
    X_labelled_scaled = class_scaler.transform(X_labelled_imp)
    y_encoded = le.transform(labelled_local["label"])
    id_lbl = labelled_local["HYRIV_ID"].values
    
    mask_train = ~np.isin(id_lbl, test_hyriv_ids)
    labelled_ids = set(id_lbl[mask_train])

    curr_X_train = X_labelled_scaled[mask_train]
    curr_y_train = y_encoded[mask_train]
    
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2, max_features=0.5, class_weight=class_weight_dict, random_state=42, n_jobs=-1, bootstrap=False)
    elif model_type == "xgb":
        model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", num_class=len(np.unique(y_encoded)), eval_metric='mlogloss', random_state=42, verbosity=0)
    elif model_type == "lgbm":
        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, num_leaves=31, subsample=0.8, colsample_bytree=0.8, class_weight="balanced", random_state=42, verbose=-1)
    elif model_type == "hist_gb":
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_depth=8, random_state=42)

    for iteration in range(num_iterations):
        model.fit(curr_X_train, curr_y_train)

        all_ids = set(rivers_features["HYRIV_ID"])
        test_ids_set = set(test_hyriv_ids)
        excluded_ids = labelled_ids.union(test_ids_set)
        unlabeled_ids = list(all_ids - excluded_ids)

        if not unlabeled_ids: break

        unlabeled_df = rivers_features[rivers_features["HYRIV_ID"].isin(unlabeled_ids)].copy()
        X_unlabeled = class_scaler.transform(class_imputer.transform(unlabeled_df[features].values))
        y_proba = model.predict_proba(X_unlabeled)
        y_pred = model.predict(X_unlabeled)

        confident_indices = []
        for i, (proba_row, pred_idx) in enumerate(zip(y_proba, y_pred)):
            class_name = le.inverse_transform([pred_idx])[0]
            class_key = class_name.split("\\")[-1].lower()
            class_thresh = class_thresholds.get(class_key, 0.95) if class_thresholds else 0.95
            if proba_row[pred_idx] >= class_thresh:
                confident_indices.append(i)

        if len(confident_indices) < min_new_labels: break

        confident_indices = np.array(confident_indices)
        X_confident = X_unlabeled[confident_indices]
        y_confident = y_pred[confident_indices]
        ids_confident = unlabeled_df.iloc[confident_indices]["HYRIV_ID"].values

        if max_per_class:
            filtered_X, filtered_y, filtered_ids = [], [], []
            counter = Counter()
            for x, y, id_ in zip(X_confident, y_confident, ids_confident):
                name = le.inverse_transform([y])[0]
                if counter[name] < max_per_class:
                    filtered_X.append(x)
                    filtered_y.append(y)
                    filtered_ids.append(id_)
                    counter[name] += 1
            X_confident = np.array(filtered_X)
            y_confident = np.array(filtered_y)
            ids_confident = np.array(filtered_ids)

        curr_X_train = np.vstack([curr_X_train, X_confident])
        curr_y_train = np.concatenate([curr_y_train, y_confident])
        labelled_ids.update(ids_confident)

    model.fit(curr_X_train, curr_y_train)
    return model

rf_thresholds = {'angular': 0.98, 'pinnate': 0.97, 'trellis': 0.88, 'parallel': 0.89, 'rectangular': 0.87, 'dendritic': 0.82, 'sub-dendritic': 0.80}
xgb_thresholds = {'angular': 0.95, 'pinnate': 0.95, 'trellis': 0.95, 'parallel': 0.90, 'rectangular': 0.90, 'dendritic': 0.85, 'sub-dendritic': 0.85}
lgbm_thresholds = {'angular': 0.96, 'pinnate': 0.96, 'trellis': 0.93, 'parallel': 0.91, 'rectangular': 0.89, 'dendritic': 0.85, 'sub-dendritic': 0.83}
hgb_thresholds = {'angular': 0.98, 'pinnate': 0.98, 'trellis': 0.98, 'parallel': 0.95, 'rectangular': 0.90, 'dendritic': 0.90, 'sub-dendritic': 0.90}

rf_model = train_models("rf", num_iterations=3, class_thresholds=rf_thresholds, min_new_labels=1500, max_per_class=2000)
xgb_model = train_models("xgb", num_iterations=3, class_thresholds=xgb_thresholds, min_new_labels=500, max_per_class=1000)
lgbm_model = train_models("lgbm", num_iterations=3, class_thresholds=lgbm_thresholds, min_new_labels=500, max_per_class=1000)
hgb_model = train_models("hist_gb", num_iterations=3, class_thresholds=hgb_thresholds, min_new_labels=500, max_per_class=1000)

models = [rf_model, xgb_model, lgbm_model, hgb_model]

print("\n--- 3. Testing pipeline on Garonne and Po ---")

def get_river_representation(df, features, scaler, imputer):
    X_raw = df[features].values
    X_raw = imputer.transform(X_raw)
    river_mean = np.mean(X_raw, axis=0).reshape(1, -1)
    return scaler.transform(river_mean)

def correct_river_class(csv_path, target_cluster_id, active_model, scaler, imputer, features):
    df = pd.read_csv(csv_path, sep=';')
    if use_feature_engineering: df = engineer_features(df)
        
    target_centroid = active_model.cluster_centers_[target_cluster_id].reshape(1, -1)
    best_dist = float('inf')
    
    for _ in range(50):
        current_vec = get_river_representation(df, features, scaler, imputer)
        current_cluster = active_model.predict(current_vec)[0]
        
        if current_cluster == target_cluster_id or len(df) < 3:
            break
            
        X_segments = scaler.transform(imputer.transform(df[features].values))
        current_sum = np.sum(X_segments, axis=0)
        n = len(df)
        
        new_means = (current_sum - X_segments) / (n - 1)
        dists = np.linalg.norm(new_means - target_centroid, axis=1)
        
        best_idx = np.argmin(dists)
        if dists[best_idx] >= best_dist:
            break
            
        best_dist = dists[best_idx]
        df = df.drop(df.index[best_idx]).reset_index(drop=True)
        
    return df

def predict_ensemble(models, X, confidence_threshold=0.5):
    rf_m, xgb_m, lgbm_m, hist_gb_m = models
    rf_proba = rf_m.predict_proba(X)
    xgb_proba = xgb_m.predict_proba(X)
    lgbm_proba = lgbm_m.predict_proba(X)
    hist_gb_proba = hist_gb_m.predict_proba(X)

    mean_proba_segments_sq = (rf_proba**2 + xgb_proba**2 + lgbm_proba**2 + hist_gb_proba**2) / 4
    mean_proba_segments = (rf_proba + xgb_proba + lgbm_proba + hist_gb_proba) / 4

    confident_mask = np.max(mean_proba_segments_sq, axis=1) >= confidence_threshold
    
    if not np.any(confident_mask):
        avg_proba, pred_mean_label, avg_entropy = None, None, None
    else:
        avg_proba = np.mean(mean_proba_segments_sq[confident_mask], axis=0)
        pred_mean_encoded = rf_m.classes_[np.argmax(avg_proba)]
        pred_mean_label = le.inverse_transform([pred_mean_encoded])[0].split("\\")[-1].lower()
        avg_entropy = entropy(avg_proba, base=2)

    return {'mean_proba': avg_proba, 'pred_mean_label': pred_mean_label, 'entropy': avg_entropy}

def composite_modification_score(original_df, modified_df, original_proba, modified_proba, target_class_index, weight_param, alpha=1.0, beta=1.0, gamma=1.0):
    n_orig, n_mod = len(original_df), len(modified_df)
    srr = (n_orig - n_mod) / n_orig if n_orig > 0 else 0
    orig_ids, mod_ids = set(original_df.index), set(modified_df.index)
    rem_length = original_df.loc[list(orig_ids - mod_ids), weight_param].sum()
    tot_length = original_df[weight_param].sum()
    wrs = rem_length / tot_length if tot_length > 0 else 0
    delta = modified_proba[target_class_index] - original_proba[target_class_index]
    delta_penalty = max(0, -delta)
    return alpha * srr + beta * wrs + gamma * delta_penalty

def correct_river_class_by_removing_segments(csv_path, target_class, models, features, scaler, imputer, confidence_threshold=0.5):
    df = pd.read_csv(csv_path, sep=';')
    if use_feature_engineering: df = engineer_features(df)
    
    rf_m = models[0]
    classes_lower = [c.split("\\")[-1].lower() for c in le.inverse_transform(rf_m.classes_)]
    class_index = classes_lower.index(target_class.lower()) if target_class.lower() in classes_lower else None
    if class_index is None: return None, None

    while True:
        X_scaled = scaler.transform(imputer.transform(df[features].values))
        res = predict_ensemble(models, X_scaled, confidence_threshold)
        
        if res['pred_mean_label'] is None: break
        if res['pred_mean_label'] == target_class.lower(): return df, res['mean_proba']
        
        rf_p, xgb_p, lgbm_p, hgb_p = [m.predict_proba(X_scaled) for m in models]
        mean_p = (rf_p**2 + xgb_p**2 + lgbm_p**2 + hgb_p**2) / 4
        
        worst_idx = np.argmin(mean_p[:, class_index])
        df = df.drop(df.index[worst_idx]).reset_index(drop=True)
        if len(df) < 3: break
            
    return None, None

inverse_kmeans_cluster_names = {v: k for k, v in kmeans_cluster_names.items()}
test_files = ["Garonne_angular.csv", "Po_pinnate.csv"]
test_items_dir = "data/test_items"

for f in test_files:
    path = os.path.join(test_items_dir, f)
    if not os.path.exists(path):
        print(f"File {path} not found. Skipping.")
        continue
        
    print(f"\n[Clustering Analysis for {f}]")
    true_class = f.split('_')[-1].replace('.csv', '').capitalize()
    if true_class not in inverse_kmeans_cluster_names:
        print(f"Target class {true_class} not mapped.")
    else:
        target_id = inverse_kmeans_cluster_names[true_class]
        target_name = kmeans_cluster_names.get(target_id)
        
        df_orig = pd.read_csv(path, sep=';')
        if use_feature_engineering: df_orig = engineer_features(df_orig)
        vec_orig = get_river_representation(df_orig, features, clust_scaler, clust_imputer)
        dist_orig = np.linalg.norm(vec_orig - kmeans.cluster_centers_[target_id])
        
        df_new = correct_river_class(path, target_id, kmeans, clust_scaler, clust_imputer, features)
        vec_new = get_river_representation(df_new, features, clust_scaler, clust_imputer)
        dist_new = np.linalg.norm(vec_new - kmeans.cluster_centers_[target_id])
        delta = dist_orig - dist_new
        print(f"Delta (Improving distance towards '{target_name}'): {delta:.4f}")

    print(f"\n[Supervised Classification & CMS Test for {f}]")
    df_river = pd.read_csv(path, sep=';')
    if use_feature_engineering: df_river = engineer_features(df_river)
    X_river = class_scaler.transform(class_imputer.transform(df_river[features].values))
    
    res = predict_ensemble(models, X_river)
    print(f"Original classification for {f}: {res['pred_mean_label']}")
    
    target_class_lower = true_class.lower()
    classes_list = [c.split("\\")[-1].lower() for c in le.inverse_transform(rf_model.classes_)]
    
    if res['pred_mean_label'] == target_class_lower:
        print(f"Classification is already correct. CMS = 0.0000")
    else:
        print(f"Pruning segments to achieve target class '{target_class_lower}'...")
        if target_class_lower in classes_list:
            target_idx = classes_list.index(target_class_lower)
            reduced_df, reduced_proba = correct_river_class_by_removing_segments(path, target_class_lower, models, features, class_scaler, class_imputer)
            if reduced_df is not None:
                cms = composite_modification_score(df_river, reduced_df, res['mean_proba'], reduced_proba, target_idx, "NET_LENGTH", 0.5, 1.5, 1.0)
                print(f"Reduction successful! New classification: {target_class_lower}")
                print(f"Composite Modification Score (CMS): {cms:.4f}")
            else:
                print("FAILURE: Unable to reduce to target class maintaining at least 3 segments.")
        else:
            print(f"Class {target_class_lower} unknown.")
    print("-" * 50)
