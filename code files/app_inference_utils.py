import os
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
def load_models():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        models = {
            "lgbm": joblib.load(os.path.join(base_dir, "adr_lightgbm_model.pkl")),
            "xgb": joblib.load(os.path.join(base_dir, "adr_xgboost_model.pkl")),
            "cat": joblib.load(os.path.join(base_dir, "adr_catboost_model.pkl")),
            "meta": joblib.load(os.path.join(base_dir, "adr_meta_model.pkl")),
            "tfidf_indication": joblib.load(os.path.join(base_dir, "tfidf_indication.pkl")),
            "tfidf_side_effect": joblib.load(os.path.join(base_dir, "tfidf_side_effect.pkl")),
            "drug_dummy_columns": joblib.load(os.path.join(base_dir, "drug_dummy_columns.pkl")),
            "feature_columns": joblib.load(os.path.join(base_dir, "feature_columns.pkl")),
            "scaler": joblib.load(os.path.join(base_dir, "scaler.pkl")),
        }

        print("✅ All models loaded successfully!")
        return models

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e


models = load_models()


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clean_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def clip_probability(value):
    value = safe_float(value)
    return max(0.0, min(1.0, value))


def safe_predict(model, features):
    try:
        pred = model.predict(features)
        if isinstance(pred, (list, tuple, np.ndarray)):
            return safe_float(pred[0])
        return safe_float(pred)
    except Exception:
        return 0.0


# -------------------------------------------------
# ENGINEERED FEATURES
# -------------------------------------------------
def compute_engineered_features(drug_name, indication_name, side_effect_name):
    drug_name = clean_text(drug_name)
    indication_name = clean_text(indication_name)
    side_effect_name = clean_text(side_effect_name)

    drug_tokens = set(drug_name.split())
    indication_tokens = set(indication_name.split())
    side_effect_tokens = set(side_effect_name.split())

    overlap_indication_side_effect = len(indication_tokens.intersection(side_effect_tokens))
    overlap_drug_side_effect = len(drug_tokens.intersection(side_effect_tokens))
    indication_length = len(indication_name)
    side_effect_length = len(side_effect_name)

    engineered = np.array([[
        overlap_indication_side_effect,
        overlap_drug_side_effect,
        indication_length,
        side_effect_length
    ]], dtype=float)

    return engineered


# -------------------------------------------------
# BUILD FEATURES
# -------------------------------------------------
def build_features(drug_name, indication_name, side_effect_name):
    drug_name = clean_text(drug_name)
    indication_name = clean_text(indication_name)
    side_effect_name = clean_text(side_effect_name)

    ind_vec = models["tfidf_indication"].transform([indication_name]).toarray()
    se_vec = models["tfidf_side_effect"].transform([side_effect_name]).toarray()

    dummy_cols = models["drug_dummy_columns"]
    drug_vec = np.zeros((1, len(dummy_cols)), dtype=float)

    drug_col_name = f"drug_{drug_name}"
    if drug_col_name in dummy_cols:
        idx = dummy_cols.index(drug_col_name)
        drug_vec[0, idx] = 1.0

    engineered = compute_engineered_features(drug_name, indication_name, side_effect_name)

    features = np.hstack([ind_vec, se_vec, drug_vec, engineered])

    try:
        features = models["scaler"].transform(features)
    except Exception as e:
        print(f"❌ Scaling error: {e}")
        raise e

    return features


# -------------------------------------------------
# BASE MODEL PREDICTIONS
# -------------------------------------------------
def get_base_predictions(features):
    xgb_pred = safe_predict(models["xgb"], features)
    lgbm_pred = safe_predict(models["lgbm"], features)
    cat_pred = safe_predict(models["cat"], features)

    return xgb_pred, lgbm_pred, cat_pred


# -------------------------------------------------
# FINAL PROBABILITY
# -------------------------------------------------
def predict_probability(features):
    try:
        xgb_pred, lgbm_pred, cat_pred = get_base_predictions(features)

        stacked = np.array([[xgb_pred, lgbm_pred, cat_pred]], dtype=float)
        final_pred = safe_predict(models["meta"], stacked)

        return clip_probability(final_pred)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 0.0


# -------------------------------------------------
# CONFIDENCE SCORE
# -------------------------------------------------
def get_confidence_score(features):
    try:
        xgb_pred, lgbm_pred, cat_pred = get_base_predictions(features)

        base_preds = np.array([xgb_pred, lgbm_pred, cat_pred], dtype=float)

        std_dev = float(np.std(base_preds))
        mean_pred = float(np.mean(base_preds))
        pred_range = float(np.max(base_preds) - np.min(base_preds))

        agreement_score = max(0.0, 1.0 - std_dev)
        spread_penalty = max(0.0, 1.0 - pred_range)
        strength_score = clip_probability(mean_pred)

        confidence = (
            0.60 * agreement_score +
            0.20 * spread_penalty +
            0.20 * strength_score
        )
        confidence = max(0.0, min(1.0, confidence))

        return confidence

    except Exception as e:
        print(f"❌ Confidence calculation error: {e}")
        return 0.0


# -------------------------------------------------
# MAIN PREDICTION FUNCTION
# -------------------------------------------------
def predict_adverse_reactions(drug_name, indication_name, side_effects_list):
    predictions = []

    if not side_effects_list:
        return predictions

    seen = set()

    for side_effect in side_effects_list:
        try:
            side_effect_clean = str(side_effect).strip()
            if not side_effect_clean:
                continue

            # avoid duplicates
            if side_effect_clean.lower() in seen:
                continue
            seen.add(side_effect_clean.lower())

            features = build_features(drug_name, indication_name, side_effect_clean)
            probability = predict_probability(features)
            confidence = get_confidence_score(features)

            predictions.append((
                side_effect_clean,
                round(probability, 4),
                round(confidence, 4)
            ))

        except Exception as e:
            print(f"❌ Error predicting for {side_effect}: {e}")
            continue

    predictions.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return predictions