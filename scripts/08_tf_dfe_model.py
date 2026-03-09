"""
TF-DFE: Final Publication Pipeline
Enhanced KNORA ensemble for somatic variant pathogenicity classification
Generates BMC Genomics-compliant figures and comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import sem, chi2_contingency
import time
import copy
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    learning_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
    brier_score_loss,
)
from sklearn.manifold import TSNE

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not available")
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude

    TDA_AVAILABLE = True
except ImportError:
    print("Warning: giotto-tda not available. Install with: pip install giotto-tda")
    TDA_AVAILABLE = False

try:
    from pyfaidx import Fasta

    PYFAIDX_AVAILABLE = True
except ImportError:
    print("Warning: pyfaidx not available. Install with: pip install pyfaidx")
    PYFAIDX_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")


class Config:
    """Configuration with optimized hyperparameters"""

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_JOBS = -1

    KNORA_K = 11
    MIN_COMPETENCE_THRESHOLD = 0.7
    TDA_N_NEIGHBORS = 75
    SEQUENCE_WINDOW = 75
    FCGR_K_VALUES = [3, 4]
    TDA_HOMOLOGY_DIMS = (0, 1)

    DATA_PATH = r"C:\Users\Admin\Downloads\Dataset-6\codes and outputs\Final code\Final_Dataset.csv"
    GENOME_PATH = r"hg19.fa"
    OUTPUT_DIR = Path("results_publication")

    LEAKAGE_COLS = ["chr", "pos", "ref", "alt", "CONSENSUS_SCORE", "TIER"]


Config.OUTPUT_DIR.mkdir(exist_ok=True)
np.random.seed(Config.RANDOM_STATE)

BMC_SINGLE_COL_WIDTH = 85 / 25.4
BMC_DOUBLE_COL_WIDTH = 170 / 25.4
BMC_DPI = 300
BMC_FONT_SIZE = 10
BMC_TITLE_SIZE = 12
BMC_LABEL_SIZE = 10

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": BMC_FONT_SIZE,
        "axes.titlesize": BMC_TITLE_SIZE,
        "axes.labelsize": BMC_LABEL_SIZE,
        "xtick.labelsize": BMC_FONT_SIZE,
        "ytick.labelsize": BMC_FONT_SIZE,
        "legend.fontsize": BMC_FONT_SIZE - 1,
        "figure.dpi": BMC_DPI,
        "savefig.dpi": BMC_DPI,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")


class FCGREncoder:
    """Frequency Chaos Game Representation encoder"""

    def __init__(self, k=4):
        self.k = k
        self.grid_size = 2**k
        self.vertices = {
            "A": (0.0, 0.0),
            "C": (0.0, 1.0),
            "G": (1.0, 1.0),
            "T": (1.0, 0.0),
        }

    def encode_sequence(self, sequence):
        sequence = sequence.upper()
        counts = np.zeros((self.grid_size, self.grid_size))
        curr_x, curr_y = 0.5, 0.5

        for nucleotide in sequence:
            if nucleotide not in self.vertices:
                continue
            vertex = self.vertices[nucleotide]
            curr_x = 0.5 * (curr_x + vertex[0])
            curr_y = 0.5 * (curr_y + vertex[1])
            row = min(int(curr_x * self.grid_size), self.grid_size - 1)
            col = min(int(curr_y * self.grid_size), self.grid_size - 1)
            counts[row, col] += 1

        if len(sequence) > 0:
            return counts.flatten() / len(sequence)
        return counts.flatten()

    def encode_variants(self, sequences):
        return np.array([self.encode_sequence(seq) for seq in sequences])


class MultiScaleFCGR:
    """Multi-scale FCGR encoder"""

    def __init__(self, k_values=[3, 4]):
        self.k_values = k_values
        self.encoders = [FCGREncoder(k=k) for k in k_values]

    def encode_variants(self, sequences):
        encoded_features = []
        for encoder in self.encoders:
            features = encoder.encode_variants(sequences)
            encoded_features.append(features)
        return np.hstack(encoded_features)


class TopologicalFeatureExtractor:
    """Extract topological features using persistent homology"""

    def __init__(self, homology_dimensions=(0, 1), n_neighbors=75):
        self.homology_dimensions = homology_dimensions
        self.n_neighbors = n_neighbors

        if TDA_AVAILABLE:
            self.vr_persistence = VietorisRipsPersistence(
                homology_dimensions=homology_dimensions, n_jobs=-1
            )
            self.persistence_entropy = PersistenceEntropy()
            self.amplitude = Amplitude(metric="persistence")

    def extract_local_tda(self, X, indices):
        if not TDA_AVAILABLE:
            return {
                "total_persistence_h0": 0.0,
                "total_persistence_h1": 0.0,
                "entropy_h0": 0.0,
                "entropy_h1": 0.0,
                "n_components_h0": 0.0,
                "n_holes_h1": 0.0,
            }

        X_local = X[indices].reshape(len(indices), -1)
        diagrams = self.vr_persistence.fit_transform([X_local])
        features = {}

        for dim in self.homology_dimensions:
            diagram = diagrams[0][diagrams[0][:, 2] == dim]
            if len(diagram) > 0:
                birth = diagram[:, 0]
                death = diagram[:, 1]
                persistence = death - birth
                features[f"total_persistence_h{dim}"] = np.sum(persistence)
                if np.sum(persistence) > 0:
                    p_normalized = persistence / np.sum(persistence)
                    entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
                    features[f"entropy_h{dim}"] = entropy
                else:
                    features[f"entropy_h{dim}"] = 0.0
                features[f"n_components_h{dim}"] = len(diagram)
            else:
                features[f"total_persistence_h{dim}"] = 0.0
                features[f"entropy_h{dim}"] = 0.0
                features[f"n_components_h{dim}"] = 0.0

        return features

    def _process_single_sample(self, X, nbrs, i, n_neighbors):
        """Process a single sample for TDA features (helper for parallel execution)"""
        distances, indices = nbrs.kneighbors([X[i]], n_neighbors=n_neighbors)
        return self.extract_local_tda(X, indices[0])

    def extract_global_tda(self, X, n_jobs=-1):
        """Extract TDA features with parallel processing

        Args:
            X: Feature matrix
            n_jobs: Number of CPU cores to use. -1 means use all available cores.
        """
        n_samples = X.shape[0]
        n_neighbors = min(self.n_neighbors, n_samples)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        nbrs.fit(X)

        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        elif n_jobs < 0:
            n_jobs = max(1, multiprocessing.cpu_count() + 1 + n_jobs)

        print(f"Extracting TDA features using {n_jobs} CPU cores...")

        # Use joblib for parallel processing with progress tracking
        tda_features_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(self._process_single_sample)(X, nbrs, i, n_neighbors)
            for i in range(n_samples)
        )

        tda_df = pd.DataFrame(tda_features_list)
        return tda_df.values


class TFDFEPreprocessor:
    """Complete preprocessing pipeline for TF-DFE"""

    def __init__(self, use_fcgr=True, use_tda=True, genome_path="hg19.fa"):
        self.leakage_cols = Config.LEAKAGE_COLS
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy="median")

        self.use_fcgr = use_fcgr
        self.use_tda = use_tda
        self.genome_path = genome_path

        self.genome = None
        if use_fcgr and PYFAIDX_AVAILABLE and Path(genome_path).exists():
            print(f"Loading reference genome: {genome_path}")
            self.genome = Fasta(genome_path)
            print(f"Genome loaded successfully")
        elif use_fcgr:
            print(f"WARNING: Genome file not found at {genome_path}")

        if use_fcgr:
            self.fcgr_encoder = MultiScaleFCGR(k_values=Config.FCGR_K_VALUES)

        if use_tda:
            self.tda_extractor = TopologicalFeatureExtractor(
                homology_dimensions=Config.TDA_HOMOLOGY_DIMS,
                n_neighbors=Config.TDA_N_NEIGHBORS,
            )

        self.feature_names = None
        self.fcgr_feature_names = None
        self.tda_feature_names = None

    def _extract_sequence_context(self, variant_df):
        """Extract genomic sequences around each variant"""
        print(f"\nExtracting genomic sequences (±{Config.SEQUENCE_WINDOW}bp)...")

        sequences = []
        failed_extractions = 0

        for idx, row in tqdm(
            variant_df.iterrows(),
            total=len(variant_df),
            desc="Sequence Extraction",
            ncols=80,
        ):
            try:
                chrom = str(row["chr"])
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"

                pos = int(row["pos"])
                start = max(0, pos - Config.SEQUENCE_WINDOW - 1)
                end = pos + Config.SEQUENCE_WINDOW

                if self.genome is not None and chrom in self.genome:
                    seq = str(self.genome[chrom][start:end]).upper()
                    if len(seq) < (2 * Config.SEQUENCE_WINDOW - 10):
                        needed_length = 2 * Config.SEQUENCE_WINDOW + len(
                            row.get("ref", "A")
                        )
                        seq = seq.ljust(needed_length, "N")
                    sequences.append(seq)
                else:
                    seq_length = 2 * Config.SEQUENCE_WINDOW + len(row.get("ref", "A"))
                    fallback_seq = "".join(
                        np.random.choice(["A", "C", "G", "T"], seq_length)
                    )
                    sequences.append(fallback_seq)
                    failed_extractions += 1

            except Exception:
                seq_length = 2 * Config.SEQUENCE_WINDOW + len(row.get("ref", "A"))
                fallback_seq = "".join(
                    np.random.choice(["A", "C", "G", "T"], seq_length)
                )
                sequences.append(fallback_seq)
                failed_extractions += 1

        print(
            f"Successfully extracted: {len(sequences) - failed_extractions}/{len(sequences)}"
        )
        return sequences

    def fit_transform(self, variant_df, target_col="LABEL_PATHOGENIC"):
        """Fit and transform training data with all feature types"""
        print("\nTF-DFE preprocessing pipeline")

        df_clean = variant_df.drop(columns=self.leakage_cols, errors="ignore")
        y = df_clean[target_col]
        X_tabular = df_clean.drop(target_col, axis=1, errors="ignore")

        num_cols = X_tabular.select_dtypes(include=["number"]).columns
        X_num = X_tabular[num_cols].copy()
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num),
            columns=num_cols,
            index=X_tabular.index,
        )

        cat_cols = X_tabular.select_dtypes(include=["object"]).columns
        X_cat = X_tabular[cat_cols].copy()
        for col in cat_cols:
            self.label_encoders[col] = LabelEncoder()
            X_cat[col] = self.label_encoders[col].fit_transform(X_cat[col].astype(str))

        X_combined = pd.concat([X_num, X_cat], axis=1)
        self.feature_names = X_combined.columns.tolist()
        X_standard = self.scaler.fit_transform(X_combined)
        print(f"Standard features: {len(self.feature_names)}")

        if self.use_fcgr:
            print("\nGenerating FCGR fractal features...")
            sequences = self._extract_sequence_context(variant_df)
            print("Encoding sequences to FCGR vectors...")
            X_fcgr = self.fcgr_encoder.encode_variants(sequences)
            total_fcgr_features = sum(4**k for k in Config.FCGR_K_VALUES)
            self.fcgr_feature_names = [f"fcgr_{i}" for i in range(total_fcgr_features)]
            print(f"FCGR features: {X_fcgr.shape[1]}")
        else:
            X_fcgr = np.array([]).reshape(X_standard.shape[0], 0)
            self.fcgr_feature_names = []

        if self.use_tda and TDA_AVAILABLE:
            print("\nGenerating TDA topological features...")
            X_tda = self.tda_extractor.extract_global_tda(X_standard)
            n_tda_features = X_tda.shape[1]
            self.tda_feature_names = [f"tda_{i}" for i in range(n_tda_features)]
            print(f"TDA features: {X_tda.shape[1]}")
        else:
            X_tda = np.array([]).reshape(X_standard.shape[0], 0)
            self.tda_feature_names = []

        all_features = [X_standard, X_fcgr, X_tda]
        X_final = np.hstack([f for f in all_features if f.shape[1] > 0])

        self.all_feature_names = (
            self.feature_names + self.fcgr_feature_names + self.tda_feature_names
        )

        print(f"\nFinal feature space: {X_final.shape[1]} features")
        print(f"  Standard: {len(self.feature_names)}")
        print(f"  FCGR:     {len(self.fcgr_feature_names)}")
        print(f"  TDA:      {len(self.tda_feature_names)}")

        return X_final, y


class SupervisedMetricLearner:
    """Learns discriminative metric space for KNORA neighbor selection"""

    def __init__(self, n_components=15):
        self.n_components = n_components
        self.lda = None
        self.pca = None
        self.mi_weights = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        print("  Learning supervised metric space...")
        self.mi_weights = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        self.mi_weights = self.mi_weights / (np.max(self.mi_weights) + 1e-10)
        X_weighted = X * self.mi_weights

        self.lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = self.lda.fit_transform(X, y)

        n_pca = min(self.n_components - 1, X.shape[1] - 1, X.shape[0] - 1)
        self.pca = PCA(n_components=n_pca, random_state=42)
        X_pca = self.pca.fit_transform(X_weighted)

        X_combined = np.hstack([X_lda, X_pca])
        self.scaler.fit(X_combined)
        self.is_fitted = True

        print(f"  Metric space: {X_combined.shape[1]}D")
        return self

    def transform(self, X):
        X_weighted = X * self.mi_weights
        X_lda = self.lda.transform(X)
        X_pca = self.pca.transform(X_weighted)
        X_combined = np.hstack([X_lda, X_pca])
        return self.scaler.transform(X_combined)


class MultiViewFeatureManager:
    """Manages multi-view feature representation"""

    def __init__(self, n_standard=18, n_fcgr=336, n_tda=6):
        self.n_standard = n_standard
        self.n_fcgr = n_fcgr
        self.n_tda = n_tda
        self.view_slices = {
            "biological": (0, n_standard),
            "fcgr": (n_standard, n_standard + n_fcgr),
            "tda": (n_standard + n_fcgr, n_standard + n_fcgr + n_tda),
            "combined": (0, n_standard + n_fcgr + n_tda),
        }


class DiverseFeatureSubspaceFactory:
    """Creates diverse base models with different feature subspaces"""

    def __init__(self, n_standard=18, n_fcgr=336, n_tda=6):
        self.view_manager = MultiViewFeatureManager(n_standard, n_fcgr, n_tda)
        self.models = {}
        self.feature_indices = {}

    def create_diverse_ensemble(self, random_state=42):
        n_std = self.view_manager.n_standard
        n_fcgr = self.view_manager.n_fcgr
        n_tda = self.view_manager.n_tda

        models = {}
        feature_indices = {}

        bio_idx = list(range(n_std))
        fcgr_idx = list(range(n_std, n_std + n_fcgr))
        tda_bio_idx = bio_idx + list(range(n_std + n_fcgr, n_std + n_fcgr + n_tda))
        bio_fcgr_idx = bio_idx + fcgr_idx
        full_idx = list(range(n_std + n_fcgr + n_tda))

        if XGBOOST_AVAILABLE:
            models["Bio_XGBoost"] = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
            feature_indices["Bio_XGBoost"] = bio_idx

        if CATBOOST_AVAILABLE:
            models["Bio_CatBoost"] = CatBoostClassifier(
                iterations=200,
                depth=5,
                learning_rate=0.05,
                random_state=random_state,
                verbose=False,
            )
            feature_indices["Bio_CatBoost"] = bio_idx

        if LGBM_AVAILABLE:
            models["FCGR_LightGBM"] = LGBMClassifier(
                n_estimators=150,
                num_leaves=63,
                learning_rate=0.05,
                feature_fraction=0.7,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            feature_indices["FCGR_LightGBM"] = fcgr_idx

        models["FCGR_RandomForest"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=random_state + 1,
            n_jobs=-1,
        )
        feature_indices["FCGR_RandomForest"] = fcgr_idx

        models["TDABio_ExtraTrees"] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=12,
            max_features="sqrt",
            random_state=random_state + 2,
            n_jobs=-1,
        )
        feature_indices["TDABio_ExtraTrees"] = tda_bio_idx

        if XGBOOST_AVAILABLE:
            models["BioFCGR_XGBoost"] = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                colsample_bytree=0.3,
                subsample=0.7,
                random_state=random_state + 3,
                n_jobs=-1,
                verbosity=0,
            )
            feature_indices["BioFCGR_XGBoost"] = bio_fcgr_idx

        models["Full_GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            max_features="sqrt",
            random_state=random_state + 4,
        )
        feature_indices["Full_GradientBoosting"] = full_idx

        models["Full_HistGradient"] = HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, learning_rate=0.05, random_state=random_state + 5
        )
        feature_indices["Full_HistGradient"] = full_idx

        self.models = models
        self.feature_indices = feature_indices

        print(f"\nCreated {len(models)} diverse base models")
        return models, feature_indices

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        trained_models = {}

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
            )

        print(f"\nTraining {len(self.models)} diverse models...")

        for name, model in tqdm(self.models.items(), desc="Training Models", ncols=80):
            indices = self.feature_indices[name]
            X_train_view = X_train[:, indices]

            try:
                model.fit(X_train_view, y_train)
                trained_models[name] = model
            except Exception as e:
                print(f"  Failed: {name} - {e}")

        self.models = trained_models
        return trained_models, X_val, y_val


class EnhancedKNORAEnsemble:
    """Enhanced KNORA-E with Supervised Metric Learning"""

    def __init__(
        self, base_classifiers, k=11, min_competence=0.7, use_metric_learning=True
    ):
        self.base_classifiers = base_classifiers
        self.k = k
        self.min_competence = min_competence
        self.use_metric_learning = use_metric_learning
        self.X_val = None
        self.y_val = None
        self.X_val_metric = None
        self.oracle_matrix = None
        self.nn_model = None
        self.metric_learner = None
        self.classifier_names = list(base_classifiers.keys())

    def fit(self, X_val, y_val):
        print("\nBuilding Enhanced KNORA-E (Best Configuration)")

        self.X_val = X_val
        self.y_val = np.array(y_val)

        n_samples = len(y_val)
        n_classifiers = len(self.base_classifiers)

        if self.use_metric_learning:
            print("\n[1/3] Supervised Metric Learning...")
            self.metric_learner = SupervisedMetricLearner(n_components=15)
            self.metric_learner.fit(X_val, y_val)
            self.X_val_metric = self.metric_learner.transform(X_val)

            print("\n[2/3] Building k-NN index (k={})...".format(self.k))
            self.nn_model = NearestNeighbors(
                n_neighbors=self.k, metric="euclidean", n_jobs=-1
            )
            self.nn_model.fit(self.X_val_metric)

        print(f"\n[3/3] Computing oracle matrix ({n_samples} x {n_classifiers})")
        self.oracle_matrix = np.zeros((n_samples, n_classifiers), dtype=int)

        for idx, (name, clf) in enumerate(self.base_classifiers.items()):
            predictions = clf.predict(X_val)
            self.oracle_matrix[:, idx] = (predictions == self.y_val).astype(int)
            accuracy = np.mean(self.oracle_matrix[:, idx])
            print(f"  {name:30} - Val Acc: {accuracy:.4f}")

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classifiers = len(self.base_classifiers)

        print(f"\nKNORA-E Inference ({n_samples} samples)...")

        all_model_preds = np.zeros((n_samples, n_classifiers))

        for idx, (name, clf) in enumerate(self.base_classifiers.items()):
            if hasattr(clf, "predict_proba"):
                try:
                    proba = clf.predict_proba(X)
                    all_model_preds[:, idx] = (
                        proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
                    )
                except:
                    all_model_preds[:, idx] = clf.predict(X).astype(float)
            else:
                all_model_preds[:, idx] = clf.predict(X).astype(float)

        if self.use_metric_learning and self.metric_learner is not None:
            X_metric = self.metric_learner.transform(X)
        else:
            X_metric = X[:, :18] if X.shape[1] > 18 else X

        distances, indices = self.nn_model.kneighbors(X_metric)

        final_predictions = np.zeros(n_samples)
        sigma = np.mean(distances) + 1e-10
        distance_weights = np.exp(-(distances**2) / (2 * sigma**2))
        distance_weights = distance_weights / (
            distance_weights.sum(axis=1, keepdims=True) + 1e-10
        )

        for i in range(n_samples):
            neighbor_indices = indices[i]
            sample_weights = distance_weights[i]

            # Ensure weights are valid (non-zero sum) to avoid ZeroDivisionError
            if np.sum(sample_weights) < 1e-10:
                sample_weights = np.ones_like(sample_weights) / len(sample_weights)

            neighbor_oracles = self.oracle_matrix[neighbor_indices]
            competence_scores = np.average(
                neighbor_oracles, axis=0, weights=sample_weights
            )

            sample_preds = all_model_preds[i]
            relative_threshold = max(
                self.min_competence, np.percentile(competence_scores, 40)
            )
            selected_mask = competence_scores >= relative_threshold

            if not np.any(selected_mask):
                top_k_indices = np.argsort(competence_scores)[
                    -max(1, n_classifiers // 2) :
                ]
                selected_mask = np.zeros(n_classifiers, dtype=bool)
                selected_mask[top_k_indices] = True

            sel_preds = sample_preds[selected_mask]
            sel_comps = competence_scores[selected_mask]

            temp = 2.0
            exp_comps = np.exp(sel_comps / temp)
            weights = exp_comps / (np.sum(exp_comps) + 1e-10)
            final_predictions[i] = np.sum(weights * sel_preds)

        return final_predictions

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class FeatureSubspaceWrapper:
    """Wrapper for models with feature subspace"""

    def __init__(self, model, feature_indices):
        self.model = model
        self.feature_indices = feature_indices

    def predict(self, X):
        X_sub = X[:, self.feature_indices]
        return self.model.predict(X_sub)

    def predict_proba(self, X):
        X_sub = X[:, self.feature_indices]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_sub)
        pred = self.model.predict(X_sub)
        return np.column_stack([1 - pred, pred])


class EnhancedTFDFEEnsemble:
    """Enhanced TF-DFE v2.0 Ensemble (Enhanced KNORA mode)"""

    def __init__(self, n_standard=18, n_fcgr=336, n_tda=6):
        self.n_standard = n_standard
        self.n_fcgr = n_fcgr
        self.n_tda = n_tda
        self.diverse_factory = None
        self.enhanced_knora = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("\nEnhanced TF-DFE v2.0 Training (Best Model)")

        n_features = X_train.shape[1]
        expected = self.n_standard + self.n_fcgr + self.n_tda

        if n_features != expected:
            print(f"Adjusting feature counts: {n_features} features")
            self.n_fcgr = n_features - self.n_standard - self.n_tda
            if self.n_fcgr < 0:
                self.n_fcgr = 0
                self.n_tda = max(0, n_features - self.n_standard)

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.15,
                stratify=y_train,
                random_state=Config.RANDOM_STATE,
            )

        self.X_val = X_val
        self.y_val = np.array(y_val)

        self.diverse_factory = DiverseFeatureSubspaceFactory(
            n_standard=self.n_standard, n_fcgr=self.n_fcgr, n_tda=self.n_tda
        )

        self.diverse_factory.create_diverse_ensemble()
        trained_models, _, _ = self.diverse_factory.train_ensemble(
            X_train, y_train, X_val, y_val
        )

        wrapped_models = {}
        for name, model in trained_models.items():
            wrapped_models[name] = FeatureSubspaceWrapper(
                model, self.diverse_factory.feature_indices[name]
            )

        self.enhanced_knora = EnhancedKNORAEnsemble(
            base_classifiers=wrapped_models,
            k=Config.KNORA_K,
            min_competence=Config.MIN_COMPETENCE_THRESHOLD,
            use_metric_learning=True,
        )
        self.enhanced_knora.fit(X_val, self.y_val)

        self.is_fitted = True
        print("\nTraining complete")

    def predict_proba(self, X):
        return self.enhanced_knora.predict_proba(X)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def plot_class_distribution(variant_df, output_dir):
    """Generate class distribution plot (Figure 1A)"""
    print("  Generating class distribution plot...")

    fig, ax = plt.subplots(figsize=(BMC_SINGLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 0.7))
    class_counts = variant_df["LABEL_PATHOGENIC"].value_counts()
    colors = ["#0077BB", "#CC3311"]

    bars = ax.bar(
        ["Benign", "Pathogenic"],
        [class_counts.get(0, 0), class_counts.get(1, 0)],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("(A) Class Distribution", fontweight="bold", loc="left")

    for bar, count in zip(bars, [class_counts.get(0, 0), class_counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=BMC_FONT_SIZE,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig1_class_distribution.png", dpi=BMC_DPI, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / "Fig1_class_distribution.tiff", dpi=BMC_DPI, bbox_inches="tight"
    )
    plt.close()


def plot_correlation_heatmap(variant_df, output_dir):
    """Generate correlation heatmap (Figure 1B)"""
    print("  Generating correlation heatmap...")

    corr_cols = [
        "REVEL_score",
        "CADD_phred",
        "SIFT_score",
        "Polyphen2_HDIV_score",
        "GERP++_RS",
        "phyloP100way_vertebrate",
        "SASA",
        "RELATIVE_SASA",
        "PLDDT_SCORE",
        "DISTANCE_TO_ACTIVE_SITE",
    ]
    available_cols = [c for c in corr_cols if c in variant_df.columns]

    if len(available_cols) >= 2:
        fig, ax = plt.subplots(
            figsize=(BMC_SINGLE_COL_WIDTH * 1.4, BMC_SINGLE_COL_WIDTH * 1.3)
        )

        short_names = {
            "REVEL_score": "REVEL",
            "CADD_phred": "CADD",
            "SIFT_score": "SIFT",
            "Polyphen2_HDIV_score": "PolyPhen2",
            "GERP++_RS": "GERP++",
            "phyloP100way_vertebrate": "phyloP",
            "SASA": "SASA",
            "RELATIVE_SASA": "Rel. SASA",
            "PLDDT_SCORE": "pLDDT",
            "DISTANCE_TO_ACTIVE_SITE": "Dist. Active",
        }

        corr_matrix = variant_df[available_cols].corr()
        corr_matrix.index = [short_names.get(c, c) for c in available_cols]
        corr_matrix.columns = [short_names.get(c, c) for c in available_cols]

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            annot_kws={"size": BMC_FONT_SIZE - 1},
        )

        ax.set_title("(B) Feature Correlation Matrix", fontweight="bold", loc="left")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig1_correlation_heatmap.png",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "Fig1_correlation_heatmap.tiff",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.close()


def plot_roc_pr_curves(y_true, y_pred_proba, model_name, output_dir):
    """Generate ROC and PR curves (Figure 1)"""
    print("  Generating ROC and PR curves...")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(BMC_DOUBLE_COL_WIDTH, BMC_DOUBLE_COL_WIDTH / 2.5)
    )

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    ax1.plot(
        fpr,
        tpr,
        linewidth=1.5,
        color="#0077BB",
        label=f"{model_name} (AUC = {auroc:.3f})",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.7, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("(A) ROC Curve", fontweight="bold", loc="left")
    ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.grid(alpha=0.3, linestyle="--")

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    ax2.plot(
        recall,
        precision,
        linewidth=1.5,
        color="#0077BB",
        label=f"{model_name} (AP = {auprc:.3f})",
    )
    baseline = np.mean(y_true)
    ax2.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"Baseline ({baseline:.3f})",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(B) Precision-Recall Curve", fontweight="bold", loc="left")
    ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "Fig1_ROC_PR_curves.png", dpi=BMC_DPI, bbox_inches="tight")
    plt.savefig(
        output_dir / "Fig1_ROC_PR_curves.tiff", dpi=BMC_DPI, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Generate confusion matrix plots (normalized and counts)"""
    print("  Generating confusion matrices...")

    cm = confusion_matrix(y_true, y_pred)
    classes = ["Benign", "Pathogenic"]

    for normalize in [False, True]:
        fig, ax = plt.subplots(
            figsize=(BMC_SINGLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 0.9)
        )

        if normalize:
            row_sums = cm.sum(axis=1)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_display = cm.astype("float") / row_sums[:, np.newaxis]
            title_suffix = "(Normalized)"
        else:
            cm_display = cm
            title_suffix = ""

        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(
            "Count" if not normalize else "Proportion", rotation=-90, va="bottom"
        )

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        ax.set_title(
            f"{model_name} Confusion Matrix {title_suffix}", fontweight="bold", pad=10
        )

        thresh = cm_display.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    text = f"{cm_display[i, j]:.1%}\n({cm[i, j]:,})"
                else:
                    text = f"{cm[i, j]:,}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=BMC_FONT_SIZE,
                )

        plt.tight_layout()
        suffix = "_normalized" if normalize else ""
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(
            output_dir / f"Fig_CM_{safe_name}{suffix}.png",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / f"Fig_CM_{safe_name}{suffix}.tiff",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.close()


def plot_tsne(X, y, title, output_dir):
    """Generate t-SNE visualization"""
    print("  Generating t-SNE visualization...")

    np.random.seed(Config.RANDOM_STATE)
    idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    X_sub = X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

    tsne = TSNE(
        n_components=2, random_state=Config.RANDOM_STATE, perplexity=30, n_iter=1000
    )
    X_embedded = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(BMC_SINGLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 0.9))

    colors = {0: "#0077BB", 1: "#CC3311"}
    for label, name in [(0, "Benign"), (1, "Pathogenic")]:
        mask = (
            (y_sub == label) if hasattr(y_sub, "values") else (np.array(y_sub) == label)
        )
        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors[label],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        title="Class", loc="best", frameon=True, fancybox=False, edgecolor="black"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "Fig_tSNE_TFDFE.png", dpi=BMC_DPI, bbox_inches="tight")
    plt.savefig(output_dir / "Fig_tSNE_TFDFE.tiff", dpi=BMC_DPI, bbox_inches="tight")
    plt.close()


def perform_shap_analysis(
    model, X_sample, feature_names, output_dir, model_name="TF-DFE"
):
    """Perform SHAP feature importance analysis"""
    if not SHAP_AVAILABLE:
        print("  SHAP analysis skipped (library not installed)")
        return None

    print("  Running SHAP analysis...")

    try:
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            print("    Using KernelExplainer (slower)...")
            background = shap.sample(X_sample, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        short_names = [
            name[:25] + "..." if len(name) > 28 else name for name in feature_names
        ]

        fig, ax = plt.subplots(
            figsize=(BMC_SINGLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 1.2)
        )
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=short_names,
            plot_type="bar",
            show=False,
            max_display=20,
            color="#0077BB",
        )
        plt.title(f"Feature Importance ({model_name})", fontweight="bold", loc="left")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_SHAP_summary.png", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_SHAP_summary.tiff", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.close()

        fig, ax = plt.subplots(
            figsize=(BMC_DOUBLE_COL_WIDTH * 0.7, BMC_SINGLE_COL_WIDTH * 1.2)
        )
        shap.summary_plot(
            shap_values, X_sample, feature_names=short_names, show=False, max_display=20
        )
        plt.title("SHAP Value Distribution", fontweight="bold", loc="left")
        plt.xlabel("SHAP Value (Impact on Pathogenicity)")
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_SHAP_beeswarm.png", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_SHAP_beeswarm.tiff", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.close()

        print("  SHAP analysis complete")
        return shap_values

    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return None


class TFDFEvaluator:
    """
    Comprehensive evaluation class for TF-DFE model analysis.
    Provides calibration analysis, statistical robustness tests,
    significance testing, and error analysis for journal publication.
    """

    @staticmethod
    def calculate_ece(y_true, y_pred_proba, n_bins=10):
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_data = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(y_pred_proba[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

                bin_data.append(
                    {
                        "bin_lower": bin_lower,
                        "bin_upper": bin_upper,
                        "avg_confidence": avg_confidence,
                        "avg_accuracy": avg_accuracy,
                        "count": np.sum(in_bin),
                        "proportion": prop_in_bin,
                        "calibration_error": np.abs(avg_accuracy - avg_confidence),
                    }
                )
            else:
                bin_data.append(
                    {
                        "bin_lower": bin_lower,
                        "bin_upper": bin_upper,
                        "avg_confidence": np.nan,
                        "avg_accuracy": np.nan,
                        "count": 0,
                        "proportion": 0,
                        "calibration_error": np.nan,
                    }
                )

        return ece, pd.DataFrame(bin_data)

    @staticmethod
    def plot_calibration_curve(
        y_true, y_pred_proba, output_dir, model_name="TF-DFE", n_bins=10
    ):
        """Generate Calibration Plot (Reliability Diagram) with histogram"""
        print("  Generating calibration curve...")

        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
        )

        ece, bin_data = TFDFEvaluator.calculate_ece(
            np.array(y_true), np.array(y_pred_proba), n_bins=n_bins
        )
        brier = brier_score_loss(y_true, y_pred_proba)

        fig = plt.figure(figsize=(BMC_SINGLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 1.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            color="#0077BB",
            linewidth=1.5,
            markersize=6,
            label=f"{model_name}",
        )

        ax1.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color="#CC3311")

        ax1.set_ylabel("Observed Probability (Positive Class)")
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.set_title(
            "Calibration Plot (Reliability Diagram)", fontweight="bold", loc="left"
        )
        ax1.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
        ax1.grid(alpha=0.3, linestyle="--")

        textstr = f"ECE = {ece:.4f}\nBrier = {brier:.4f}"
        props = dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
        ax1.text(
            0.95,
            0.05,
            textstr,
            transform=ax1.transAxes,
            fontsize=BMC_FONT_SIZE,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        ax1.set_xticklabels([])

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.hist(
            y_pred_proba,
            bins=n_bins,
            range=(0, 1),
            color="#0077BB",
            edgecolor="black",
            alpha=0.7,
            linewidth=0.5,
        )
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_xlim([-0.02, 1.02])

        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_Calibration_Curve.png", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_Calibration_Curve.tiff", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.close()

        bin_data.to_csv(output_dir / "calibration_bins.csv", index=False)

        print(f"    ECE: {ece:.6f}")
        print(f"    Brier Score: {brier:.6f}")
        print(f"    Saved: Fig_Calibration_Curve.png, calibration_bins.csv")

        return ece, brier, bin_data

    @staticmethod
    def perform_repeated_cv(
        X,
        y,
        model_factory_fn,
        n_splits=5,
        n_repeats=2,
        n_standard=10,
        n_fcgr=336,
        n_tda=6,
        output_dir=None,
    ):
        """Perform Repeated Stratified K-Fold Cross-Validation for robust statistics"""
        print(f"\n  Running {n_repeats}x{n_splits}-fold Repeated Stratified CV...")

        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=Config.RANDOM_STATE
        )

        cv_results = []
        fold_idx = 0

        y_arr = np.array(y) if hasattr(y, "values") else y

        for train_idx, test_idx in tqdm(
            rskf.split(X, y_arr),
            total=n_splits * n_repeats,
            desc="CV Progress",
            ncols=80,
        ):
            fold_idx += 1
            repeat_num = (fold_idx - 1) // n_splits + 1
            fold_num = (fold_idx - 1) % n_splits + 1

            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y_arr[train_idx], y_arr[test_idx]

            model = model_factory_fn(n_standard, n_fcgr, n_tda)
            model.fit(X_train_cv, y_train_cv)

            y_pred_cv = model.predict(X_test_cv)
            y_pred_proba_cv = model.predict_proba(X_test_cv)

            fold_metrics = {
                "repeat": repeat_num,
                "fold": fold_num,
                "mcc": matthews_corrcoef(y_test_cv, y_pred_cv),
                "auprc": average_precision_score(y_test_cv, y_pred_proba_cv),
                "auroc": roc_auc_score(y_test_cv, y_pred_proba_cv),
                "f1": f1_score(y_test_cv, y_pred_cv, zero_division=0),
                "precision": precision_score(y_test_cv, y_pred_cv, zero_division=0),
                "recall": recall_score(y_test_cv, y_pred_cv, zero_division=0),
                "accuracy": accuracy_score(y_test_cv, y_pred_cv),
                "brier": brier_score_loss(y_test_cv, y_pred_proba_cv),
            }
            cv_results.append(fold_metrics)

        cv_df = pd.DataFrame(cv_results)

        summary_stats = {}
        metrics_to_summarize = [
            "mcc",
            "auprc",
            "auroc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "brier",
        ]

        for metric in metrics_to_summarize:
            values = cv_df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            ci_95 = stats.t.ppf(0.975, n - 1) * (std_val / np.sqrt(n))

            summary_stats[metric] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": ci_95,
                "ci_lower": mean_val - ci_95,
                "ci_upper": mean_val + ci_95,
                "formatted": f"{mean_val:.4f} ± {ci_95:.4f}",
            }

        if output_dir:
            cv_df.to_csv(output_dir / "cv_metrics.csv", index=False)

            summary_df = pd.DataFrame(
                [
                    {
                        "metric": metric,
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "ci_95": stats["ci_95"],
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "formatted": stats["formatted"],
                    }
                    for metric, stats in summary_stats.items()
                ]
            )
            summary_df.to_csv(output_dir / "cv_summary_stats.csv", index=False)

            print(f"    Saved: cv_metrics.csv, cv_summary_stats.csv")

        print(f"\n  CV Results Summary (Mean ± 95% CI):")
        print(f"    MCC:    {summary_stats['mcc']['formatted']}")
        print(f"    AUPRC:  {summary_stats['auprc']['formatted']}")
        print(f"    AUROC:  {summary_stats['auroc']['formatted']}")
        print(f"    F1:     {summary_stats['f1']['formatted']}")

        return cv_df, summary_stats

    @staticmethod
    def plot_cv_distribution(cv_df, output_dir, metrics=["mcc", "auprc"]):
        """Generate Box/Violin plot showing the spread of metrics across CV folds"""
        print("  Generating CV distribution plot...")

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(BMC_SINGLE_COL_WIDTH * len(metrics), BMC_SINGLE_COL_WIDTH * 0.8),
        )

        if len(metrics) == 1:
            axes = [axes]

        colors = ["#0077BB", "#EE7733", "#009988", "#CC3311"]
        metric_names = {
            "mcc": "MCC",
            "auprc": "AUPRC",
            "auroc": "AUROC",
            "f1": "F1-Score",
        }

        for i, metric in enumerate(metrics):
            ax = axes[i]

            parts = ax.violinplot(
                cv_df[metric], positions=[1], showmeans=True, showextrema=True
            )

            for pc in parts["bodies"]:
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.6)

            bp = ax.boxplot(
                cv_df[metric],
                positions=[1],
                widths=0.15,
                patch_artist=True,
                showfliers=True,
            )
            bp["boxes"][0].set_facecolor("white")
            bp["boxes"][0].set_alpha(0.8)

            jitter = np.random.uniform(-0.05, 0.05, len(cv_df))
            ax.scatter(
                np.ones(len(cv_df)) + jitter,
                cv_df[metric],
                alpha=0.5,
                s=20,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

            mean_val = cv_df[metric].mean()
            std_val = cv_df[metric].std()
            n = len(cv_df)
            ci_95 = stats.t.ppf(0.975, n - 1) * (std_val / np.sqrt(n))

            ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, alpha=0.7)

            ax.set_ylabel(metric_names.get(metric, metric.upper()))
            ax.set_title(
                f"{metric_names.get(metric, metric.upper())}\n{mean_val:.4f} ± {ci_95:.4f}",
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.suptitle("Cross-Validation Metric Distributions", fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_CV_Distribution.png", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_CV_Distribution.tiff", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.close()

        print(f"    Saved: Fig_CV_Distribution.png")

    @staticmethod
    def perform_mcnemar_test(
        y_true,
        y_pred_model1,
        y_pred_model2,
        model1_name="Enhanced Model",
        model2_name="Baseline",
        output_dir=None,
    ):
        """Perform McNemar's Test for statistical significance between two models"""
        print(f"\n  Performing McNemar's Test: {model1_name} vs {model2_name}")

        y_true = np.array(y_true)
        y_pred_model1 = np.array(y_pred_model1)
        y_pred_model2 = np.array(y_pred_model2)

        correct_1 = y_pred_model1 == y_true
        correct_2 = y_pred_model2 == y_true

        a = np.sum(correct_1 & correct_2)
        b = np.sum(correct_1 & ~correct_2)
        c = np.sum(~correct_1 & correct_2)
        d = np.sum(~correct_1 & ~correct_2)

        print(f"\n    Contingency Table:")
        print(f"    {'':20} {model2_name:>15} {model2_name:>15}")
        print(f"    {'':20} {'Correct':>15} {'Wrong':>15}")
        print(f"    {model1_name:20} {a:>15,} {b:>15,}")
        print(f"    {model1_name + ' Wrong':20} {c:>15,} {d:>15,}")

        if b + c == 0:
            print(
                "    WARNING: No disagreements between models. McNemar's test not applicable."
            )
            statistic, p_value = np.nan, np.nan
            interpretation = "No disagreements - test not applicable"
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

            if b + c < 25:
                p_value_exact = 2 * min(
                    stats.binom.cdf(min(b, c), b + c, 0.5),
                    1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5),
                )
                p_value = p_value_exact
                print(f"    (Using exact binomial test due to small sample)")

            if p_value < 0.001:
                interpretation = "Highly significant difference (p < 0.001)"
                sig_symbol = "***"
            elif p_value < 0.01:
                interpretation = "Very significant difference (p < 0.01)"
                sig_symbol = "**"
            elif p_value < 0.05:
                interpretation = "Significant difference (p < 0.05)"
                sig_symbol = "*"
            else:
                interpretation = "No significant difference (p >= 0.05)"
                sig_symbol = "ns"

        acc_model1 = np.mean(correct_1)
        acc_model2 = np.mean(correct_2)
        acc_diff = acc_model1 - acc_model2

        result = {
            "model1": model1_name,
            "model2": model2_name,
            "model1_accuracy": acc_model1,
            "model2_accuracy": acc_model2,
            "accuracy_difference": acc_diff,
            "both_correct": a,
            "model1_only_correct": b,
            "model2_only_correct": c,
            "both_wrong": d,
            "statistic": statistic,
            "p_value": p_value,
            "significance": sig_symbol if "sig_symbol" in dir() else "N/A",
            "interpretation": interpretation,
        }

        print(f"\n    Results:")
        print(
            f"    McNemar Statistic: {statistic:.4f}"
            if not np.isnan(statistic)
            else "    McNemar Statistic: N/A"
        )
        print(
            f"    P-value: {p_value:.6f}"
            if not np.isnan(p_value)
            else "    P-value: N/A"
        )
        print(f"    {interpretation}")
        print(f"\n    {model1_name} Accuracy: {acc_model1:.4f}")
        print(f"    {model2_name} Accuracy: {acc_model2:.4f}")
        print(f"    Accuracy Improvement: {acc_diff:+.4f}")

        if output_dir:
            result_df = pd.DataFrame([result])
            result_df.to_csv(output_dir / "significance_test_mcnemar.csv", index=False)
            print(f"    Saved: significance_test_mcnemar.csv")

        return result

    @staticmethod
    def generate_misclassification_report(
        variant_df, y_true, y_pred, y_pred_proba, test_indices, output_dir
    ):
        """Generate detailed report of misclassified samples for error analysis"""
        print("  Generating misclassification report...")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        misclassified_mask = y_true != y_pred
        misclass_indices = np.where(misclassified_mask)[0]

        original_indices = test_indices[misclass_indices]

        coord_cols = ["chr", "pos", "ref", "alt"]
        available_coord_cols = [c for c in coord_cols if c in variant_df.columns]

        misclass_records = []
        for i, orig_idx in enumerate(original_indices):
            local_idx = misclass_indices[i]
            record = {
                "test_index": local_idx,
                "original_index": orig_idx,
                "y_true": int(y_true[local_idx]),
                "y_pred": int(y_pred[local_idx]),
                "y_pred_proba": float(y_pred_proba[local_idx]),
                "error_type": "FP" if y_pred[local_idx] == 1 else "FN",
                "confidence": abs(y_pred_proba[local_idx] - 0.5) * 2,
            }

            for col in available_coord_cols:
                record[col] = variant_df.iloc[orig_idx][col]

            misclass_records.append(record)

        misclass_df = pd.DataFrame(misclass_records)

        misclass_df = misclass_df.sort_values("confidence", ascending=False)

        n_fp = np.sum(misclass_df["error_type"] == "FP")
        n_fn = np.sum(misclass_df["error_type"] == "FN")

        print(f"    Total misclassifications: {len(misclass_df)}")
        print(f"    False Positives (FP): {n_fp}")
        print(f"    False Negatives (FN): {n_fn}")

        misclass_df.to_csv(output_dir / "misclassifications.csv", index=False)
        print(f"    Saved: misclassifications.csv")

        return misclass_df

    @staticmethod
    def aggregate_shap_by_category(
        shap_values, feature_names, n_standard, n_fcgr, n_tda, output_dir
    ):
        """Aggregate SHAP values by feature category (Biological, FCGR, TDA)"""
        print("  Aggregating SHAP values by feature category...")

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        n_features = len(mean_abs_shap)

        category_shap = {
            "Standard/Biological": [],
            "Fractal (FCGR)": [],
            "Topological (TDA)": [],
        }

        feature_category_map = []

        for i, fname in enumerate(feature_names):
            if i < n_standard or fname.startswith(
                (
                    "REVEL",
                    "CADD",
                    "SIFT",
                    "Polyphen",
                    "GERP",
                    "phyloP",
                    "phastCons",
                    "DANN",
                    "fathmm",
                    "MutationTaster",
                    "MPC",
                )
            ):
                category = "Standard/Biological"
            elif "fcgr" in fname.lower() or i < n_standard + n_fcgr:
                category = "Fractal (FCGR)"
            elif (
                "tda" in fname.lower()
                or "persistence" in fname.lower()
                or "entropy" in fname.lower()
            ):
                category = "Topological (TDA)"
            else:
                if i < n_standard:
                    category = "Standard/Biological"
                elif i < n_standard + n_fcgr:
                    category = "Fractal (FCGR)"
                else:
                    category = "Topological (TDA)"

            category_shap[category].append(mean_abs_shap[i])
            feature_category_map.append(
                {
                    "feature_name": fname,
                    "category": category,
                    "mean_abs_shap": mean_abs_shap[i],
                }
            )

        aggregated_results = []
        for category, values in category_shap.items():
            if len(values) > 0:
                aggregated_results.append(
                    {
                        "category": category,
                        "n_features": len(values),
                        "mean_abs_shap": np.mean(values),
                        "sum_abs_shap": np.sum(values),
                        "std_abs_shap": np.std(values),
                        "max_abs_shap": np.max(values),
                        "relative_importance": np.sum(values)
                        / np.sum(mean_abs_shap)
                        * 100,
                    }
                )

        aggregated_df = pd.DataFrame(aggregated_results)
        aggregated_df = aggregated_df.sort_values("mean_abs_shap", ascending=False)

        feature_df = pd.DataFrame(feature_category_map)
        feature_df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)
        aggregated_df.to_csv(
            output_dir / "feature_importance_aggregated.csv", index=False
        )

        print(f"\n    Category-Level Importance:")
        for _, row in aggregated_df.iterrows():
            print(
                f"    {row['category']:25} - Mean |SHAP|: {row['mean_abs_shap']:.6f} "
                f"({row['relative_importance']:.1f}% of total)"
            )

        print(
            f"    Saved: feature_importance_aggregated.csv, feature_importance_detailed.csv"
        )

        return aggregated_df, feature_df

    @staticmethod
    def plot_feature_importance_stacked(aggregated_df, output_dir):
        """Generate stacked/grouped bar chart showing contribution of feature categories"""
        print("  Generating feature importance stacked bar chart...")

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(BMC_DOUBLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 0.7)
        )

        colors = {
            "Standard/Biological": "#0077BB",
            "Fractal (FCGR)": "#EE7733",
            "Topological (TDA)": "#009988",
        }

        categories = aggregated_df["category"].tolist()
        bar_colors = [colors.get(cat, "#888888") for cat in categories]

        bars = ax1.barh(
            categories,
            aggregated_df["mean_abs_shap"],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Mean |SHAP Value|")
        ax1.set_title("(A) Feature Category Importance", fontweight="bold", loc="left")
        ax1.grid(axis="x", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, aggregated_df["mean_abs_shap"]):
            ax1.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=BMC_FONT_SIZE - 1,
            )

        wedges, texts, autotexts = ax2.pie(
            aggregated_df["relative_importance"],
            labels=categories,
            colors=bar_colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.02] * len(categories),
            wedgeprops=dict(linewidth=1, edgecolor="white"),
        )
        ax2.set_title("(B) Relative Contribution", fontweight="bold")

        for autotext in autotexts:
            autotext.set_fontsize(BMC_FONT_SIZE)
            autotext.set_fontweight("bold")

        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_Feature_Importance_Stacked.png",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "Fig_Feature_Importance_Stacked.tiff",
            dpi=BMC_DPI,
            bbox_inches="tight",
        )
        plt.close()

        print(f"    Saved: Fig_Feature_Importance_Stacked.png")

    @staticmethod
    def generate_learning_curve(
        X,
        y,
        model_factory_fn,
        n_standard,
        n_fcgr,
        n_tda,
        output_dir,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        """Generate learning curve showing performance vs training data size"""
        print(f"\n  Generating Learning Curve...")
        print(f"    Training sizes: {[f'{s*100:.0f}%' for s in train_sizes]}")

        y_arr = np.array(y) if hasattr(y, "values") else y

        results = []

        for train_size in tqdm(train_sizes, desc="Learning Curve", ncols=80):
            if train_size < 1.0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_arr,
                    train_size=train_size,
                    stratify=y_arr,
                    random_state=Config.RANDOM_STATE,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_arr,
                    test_size=0.2,
                    stratify=y_arr,
                    random_state=Config.RANDOM_STATE,
                )

            model = model_factory_fn(n_standard, n_fcgr, n_tda)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            results.append(
                {
                    "train_size_pct": train_size * 100,
                    "n_train_samples": len(X_train),
                    "n_test_samples": len(X_test),
                    "mcc": matthews_corrcoef(y_test, y_pred),
                    "auprc": average_precision_score(y_test, y_pred_proba),
                    "auroc": roc_auc_score(y_test, y_pred_proba),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "accuracy": accuracy_score(y_test, y_pred),
                }
            )

        results_df = pd.DataFrame(results)

        fig, axes = plt.subplots(
            1, 2, figsize=(BMC_DOUBLE_COL_WIDTH, BMC_SINGLE_COL_WIDTH * 0.7)
        )

        ax1 = axes[0]
        ax1.plot(
            results_df["train_size_pct"],
            results_df["mcc"],
            "o-",
            color="#0077BB",
            linewidth=1.5,
            markersize=6,
            label="MCC",
        )
        ax1.plot(
            results_df["train_size_pct"],
            results_df["f1"],
            "s--",
            color="#EE7733",
            linewidth=1.5,
            markersize=6,
            label="F1-Score",
        )
        ax1.set_xlabel("Training Data Size (%)")
        ax1.set_ylabel("Score")
        ax1.set_title("(A) Learning Curve - MCC & F1", fontweight="bold", loc="left")
        ax1.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
        ax1.grid(alpha=0.3, linestyle="--")
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 1.05])

        ax2 = axes[1]
        ax2.plot(
            results_df["train_size_pct"],
            results_df["auprc"],
            "o-",
            color="#009988",
            linewidth=1.5,
            markersize=6,
            label="AUPRC",
        )
        ax2.plot(
            results_df["train_size_pct"],
            results_df["auroc"],
            "s--",
            color="#CC3311",
            linewidth=1.5,
            markersize=6,
            label="AUROC",
        )
        ax2.set_xlabel("Training Data Size (%)")
        ax2.set_ylabel("Score")
        ax2.set_title(
            "(B) Learning Curve - AUPRC & AUROC", fontweight="bold", loc="left"
        )
        ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="black")
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_Learning_Curve.png", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_Learning_Curve.tiff", dpi=BMC_DPI, bbox_inches="tight"
        )
        plt.close()

        results_df.to_csv(output_dir / "learning_curve_data.csv", index=False)

        print(f"\n    Learning Curve Results:")
        for _, row in results_df.iterrows():
            print(
                f"    {row['train_size_pct']:5.0f}% ({row['n_train_samples']:,} samples): "
                f"MCC={row['mcc']:.4f}, AUPRC={row['auprc']:.4f}"
            )

        print(f"    Saved: Fig_Learning_Curve.png, learning_curve_data.csv")

        return results_df


class BaselineEnsemble:
    """Simple Baseline Model: Ensemble of 8 Tree models on 18 Biological features"""

    def __init__(self, n_biological_features=18, random_state=42):
        self.n_biological_features = n_biological_features
        self.random_state = random_state
        self.models = {}
        self.is_fitted = False

    def _create_base_models(self):
        """Create 8 diverse tree-based models"""
        models = {
            "RF_1": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "RF_2": RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state + 1,
                n_jobs=-1,
            ),
            "ET_1": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state + 2,
                n_jobs=-1,
            ),
            "ET_2": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=self.random_state + 3,
                n_jobs=-1,
            ),
            "GB_1": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=self.random_state + 4
            ),
            "GB_2": GradientBoostingClassifier(
                n_estimators=100, max_depth=7, random_state=self.random_state + 5
            ),
            "HGB_1": HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, random_state=self.random_state + 6
            ),
            "HGB_2": HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, random_state=self.random_state + 7
            ),
        }
        return models

    def fit(self, X, y):
        """Train baseline ensemble on biological features only"""
        X_bio = X[:, : self.n_biological_features]

        self.models = self._create_base_models()

        for name, model in tqdm(
            self.models.items(), desc="Training Baseline", ncols=80
        ):
            model.fit(X_bio, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Get ensemble probability predictions"""
        X_bio = X[:, : self.n_biological_features]

        probas = []
        for name, model in self.models.items():
            proba = model.predict_proba(X_bio)[:, 1]
            probas.append(proba)

        return np.mean(probas, axis=0)

    def predict(self, X, threshold=0.5):
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute comprehensive metrics"""
    metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
        metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["brier"] = brier_score_loss(y_true, y_pred_proba)

    return metrics


def print_metrics_report(metrics, model_name="Model"):
    """Print formatted metrics report"""
    print(f"\n{model_name} - Performance Metrics")
    print(f"\nPrimary Metrics:")
    print(f"  Matthews Correlation Coefficient (MCC): {metrics.get('mcc', 0):.6f}")
    print(f"  AUPRC (Precision-Recall AUC):           {metrics.get('auprc', 0):.6f}")
    print(f"  Brier Score (Calibration):              {metrics.get('brier', 0):.6f}")

    print(f"\nSecondary Metrics:")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.6f}")
    print(f"  Precision:  {metrics.get('precision', 0):.6f}")
    print(f"  Recall:     {metrics.get('recall', 0):.6f}")
    print(f"  F1-Score:   {metrics.get('f1', 0):.6f}")
    print(f"  AUROC:      {metrics.get('auroc', 0):.6f}")
    print(f"  Cohen's κ:  {metrics.get('kappa', 0):.6f}")


def run_final_model():
    """
    Run the best performing Enhanced KNORA model on the full dataset.
    Generates all visualization figures and metrics for publication.
    """
    start_time = time.time()

    print("\nTF-DFE: Final Publication Run")
    print("Best Model: Enhanced KNORA")
    print(f"\nBest Hyperparameters:")
    print(f"  knora_k:        {Config.KNORA_K}")
    print(f"  min_competence: {Config.MIN_COMPETENCE_THRESHOLD}")
    print(f"  tda_neighbors:  {Config.TDA_N_NEIGHBORS}")
    print(f"  sequence_window: {Config.SEQUENCE_WINDOW}")
    print(f"\nOutput Directory: {Config.OUTPUT_DIR}")

    print("\nStep 1: Load Data & Exploratory Data Analysis")

    if not Path(Config.DATA_PATH).exists():
        print(f"ERROR: Data file not found at {Config.DATA_PATH}")
        return

    print(f"\nLoading: {Config.DATA_PATH}")
    variant_df = pd.read_csv(Config.DATA_PATH)
    print(f"Loaded {len(variant_df):,} variants")

    TEST_SIZE_LIMIT = 500000000

    if len(variant_df) > TEST_SIZE_LIMIT:
        print(f"\nTEST MODE: Sampling {TEST_SIZE_LIMIT} variants for faster testing...")

        pathogenic_samples = variant_df[variant_df["LABEL_PATHOGENIC"] == 1].sample(
            TEST_SIZE_LIMIT // 2, random_state=Config.RANDOM_STATE
        )
        benign_samples = variant_df[variant_df["LABEL_PATHOGENIC"] == 0].sample(
            TEST_SIZE_LIMIT // 2, random_state=Config.RANDOM_STATE
        )

        variant_df = pd.concat([pathogenic_samples, benign_samples]).reset_index(
            drop=True
        )
        print(
            f"Dataset reduced to {len(variant_df):,} samples (stratified 50/50 split)"
        )

    if "LABEL_PATHOGENIC" in variant_df.columns:
        target_dist = variant_df["LABEL_PATHOGENIC"].value_counts().to_dict()
        print(f"\nClass Distribution:")
        for label, count in sorted(target_dist.items()):
            label_name = "Pathogenic" if label == 1 else "Benign"
            pct = count / len(variant_df) * 100
            print(f"  {label_name:12}: {count:6,} ({pct:5.2f}%)")

    print("\nGenerating EDA Figures (Figure 1)...")
    plot_class_distribution(variant_df, Config.OUTPUT_DIR)
    plot_correlation_heatmap(variant_df, Config.OUTPUT_DIR)
    print("EDA figures saved")

    print("\nStep 2: Feature Engineering")

    preprocessor = TFDFEPreprocessor(
        use_fcgr=True, use_tda=True, genome_path=Config.GENOME_PATH
    )

    X, y = preprocessor.fit_transform(variant_df)

    n_standard = len(preprocessor.feature_names)
    n_fcgr = len(preprocessor.fcgr_feature_names)
    n_tda = len(preprocessor.tda_feature_names)

    print(f"\nFeature engineering complete")
    print(f"  Total features: {X.shape[1]}")

    print("\nStep 3: Train Best Model (Enhanced KNORA)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE
    )

    print(f"\nData Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    model = EnhancedTFDFEEnsemble(n_standard=n_standard, n_fcgr=n_fcgr, n_tda=n_tda)

    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"\nTraining complete ({train_time/60:.2f} minutes)")

    print("\nGenerating predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    print_metrics_report(metrics, model_name="Enhanced KNORA (TF-DFE)")

    print("\nStep 4: Generate Publication Figures")

    plot_roc_pr_curves(y_test, y_pred_proba, "Enhanced KNORA", Config.OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_pred, "TF-DFE", Config.OUTPUT_DIR)
    plot_tsne(X_test, y_test, "TF-DFE Feature Space Separation", Config.OUTPUT_DIR)

    if SHAP_AVAILABLE:
        shap_model = None
        if hasattr(model, "diverse_factory") and model.diverse_factory is not None:
            for name, m in model.diverse_factory.models.items():
                if (
                    hasattr(m, "estimators_")
                    or "XGB" in name
                    or "LGB" in name
                    or "CatBoost" in name
                ):
                    shap_model = m
                    shap_indices = model.diverse_factory.feature_indices[name]
                    shap_feature_names = [
                        preprocessor.all_feature_names[i] for i in shap_indices
                    ]
                    X_shap = X_test[: min(500, len(X_test)), shap_indices]
                    break

        if shap_model:
            perform_shap_analysis(
                shap_model, X_shap, shap_feature_names, Config.OUTPUT_DIR, "TF-DFE"
            )
        else:
            print("  SHAP analysis skipped (no compatible model found)")

    print("\nAll figures generated")

    print("\nStep 4A: Calibration Analysis")

    y_test_arr = np.array(y_test) if hasattr(y_test, "values") else y_test
    ece, brier, calibration_bins = TFDFEvaluator.plot_calibration_curve(
        y_test_arr,
        y_pred_proba,
        Config.OUTPUT_DIR,
        model_name="Enhanced KNORA (TF-DFE)",
        n_bins=10,
    )

    metrics["ece"] = ece

    print("\nStep 4B: Statistical Significance Test (McNemar's)")

    print("\n  Training Baseline Model for comparison...")
    baseline_model = BaselineEnsemble(
        n_biological_features=n_standard, random_state=Config.RANDOM_STATE
    )
    baseline_model.fit(X_train, y_train)

    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_proba_baseline = baseline_model.predict_proba(X_test)

    baseline_metrics = compute_metrics(
        y_test_arr, y_pred_baseline, y_pred_proba_baseline
    )
    print(f"\n  Baseline Model Metrics:")
    print(f"    MCC:    {baseline_metrics['mcc']:.6f}")
    print(f"    AUPRC:  {baseline_metrics['auprc']:.6f}")
    print(f"    F1:     {baseline_metrics['f1']:.6f}")

    mcnemar_result = TFDFEvaluator.perform_mcnemar_test(
        y_test_arr,
        y_pred,
        y_pred_baseline,
        model1_name="Enhanced KNORA",
        model2_name="Baseline Ensemble",
        output_dir=Config.OUTPUT_DIR,
    )

    print("\nStep 4C: Error Analysis")

    y_array = np.array(y) if hasattr(y, "values") else y
    _, _, _, _, train_indices, test_indices = train_test_split(
        X,
        y_array,
        np.arange(len(y)),
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE,
    )

    misclass_df = TFDFEvaluator.generate_misclassification_report(
        variant_df, y_test_arr, y_pred, y_pred_proba, test_indices, Config.OUTPUT_DIR
    )

    print("\nStep 4D: Aggregated Feature Importance by Category")

    if SHAP_AVAILABLE and shap_model:
        try:
            print("  Computing SHAP values for category aggregation...")

            sample_size = min(1000, len(X_test))
            X_sample_full = X_test[:sample_size]

            if "Full_GradientBoosting" in model.diverse_factory.models:
                agg_model = model.diverse_factory.models["Full_GradientBoosting"]
                agg_indices = model.diverse_factory.feature_indices[
                    "Full_GradientBoosting"
                ]
            elif "Full_HistGradient" in model.diverse_factory.models:
                agg_model = model.diverse_factory.models["Full_HistGradient"]
                agg_indices = model.diverse_factory.feature_indices["Full_HistGradient"]
            else:
                agg_model = shap_model
                agg_indices = shap_indices

            X_sample_subset = X_sample_full[:, agg_indices]
            agg_feature_names = [preprocessor.all_feature_names[i] for i in agg_indices]

            if hasattr(agg_model, "estimators_"):
                explainer = shap.TreeExplainer(agg_model)
            else:
                background = shap.sample(X_sample_subset, 50)
                explainer = shap.KernelExplainer(agg_model.predict_proba, background)

            agg_shap_values = explainer.shap_values(X_sample_subset)

            if isinstance(agg_shap_values, list) and len(agg_shap_values) == 2:
                agg_shap_values = agg_shap_values[1]
            elif (
                isinstance(agg_shap_values, np.ndarray)
                and len(agg_shap_values.shape) == 3
            ):
                agg_shap_values = agg_shap_values[:, :, 1]

            aggregated_df, feature_df = TFDFEvaluator.aggregate_shap_by_category(
                agg_shap_values,
                agg_feature_names,
                n_standard,
                n_fcgr,
                n_tda,
                Config.OUTPUT_DIR,
            )

            TFDFEvaluator.plot_feature_importance_stacked(
                aggregated_df, Config.OUTPUT_DIR
            )

        except Exception as e:
            print(f"  Aggregated SHAP analysis failed: {e}")
    else:
        print("  Skipped (SHAP not available or no compatible model)")

    print("\nStep 4E: Learning Curve Analysis")

    def model_factory_fn(n_std, n_fcg, n_td):
        return EnhancedTFDFEEnsemble(n_standard=n_std, n_fcgr=n_fcg, n_tda=n_td)

    if len(X) > 50000:
        print(
            f"  Dataset size ({len(X):,}) is large. Using subset for learning curve..."
        )
        lc_sample_size = min(50000, len(X))
        lc_indices = np.random.choice(len(X), lc_sample_size, replace=False)
        X_lc = X[lc_indices]
        y_lc = y_array[lc_indices]
    else:
        X_lc = X
        y_lc = y_array

    learning_curve_df = TFDFEvaluator.generate_learning_curve(
        X_lc,
        y_lc,
        model_factory_fn,
        n_standard,
        n_fcgr,
        n_tda,
        Config.OUTPUT_DIR,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    print("\nStep 4F: Repeated Stratified K-Fold Cross-Validation")

    if len(X) > 100000:
        print(f"  Dataset size ({len(X):,}) is large. Using subset for CV analysis...")
        cv_sample_size = min(50000, len(X))
        cv_indices = np.random.choice(len(X), cv_sample_size, replace=False)
        X_cv = X[cv_indices]
        y_cv = y_array[cv_indices]
    else:
        X_cv = X
        y_cv = y_array

    cv_df, cv_summary = TFDFEvaluator.perform_repeated_cv(
        X_cv,
        y_cv,
        model_factory_fn,
        n_splits=5,
        n_repeats=2,
        n_standard=n_standard,
        n_fcgr=n_fcgr,
        n_tda=n_tda,
        output_dir=Config.OUTPUT_DIR,
    )

    TFDFEvaluator.plot_cv_distribution(
        cv_df, Config.OUTPUT_DIR, metrics=["mcc", "auprc"]
    )

    print("\nStep 5: Save Reports")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Config.OUTPUT_DIR / "final_metrics.csv", index=False)
    print(f"  {Config.OUTPUT_DIR / 'final_metrics.csv'}")

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.values if hasattr(y_test, "values") else y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }
    )
    predictions_df.to_csv(Config.OUTPUT_DIR / "final_predictions.csv", index=False)
    print(f"  {Config.OUTPUT_DIR / 'final_predictions.csv'}")

    with open(Config.OUTPUT_DIR / "final_report.txt", "w") as report_file:
        report_file.write("TF-DFE Final Publication Report\n")
        report_file.write("BMC Genomics Manuscript Submission\n")
        report_file.write("\n")

        report_file.write("Best Hyperparameters\n")
        report_file.write(f"knora_k: {Config.KNORA_K}\n")
        report_file.write(f"min_competence: {Config.MIN_COMPETENCE_THRESHOLD}\n")
        report_file.write(f"tda_neighbors: {Config.TDA_N_NEIGHBORS}\n")
        report_file.write(f"sequence_window: {Config.SEQUENCE_WINDOW}\n\n")

        report_file.write("Dataset Information\n")
        report_file.write(f"Total variants: {len(variant_df):,}\n")
        report_file.write(f"Training samples: {len(X_train):,}\n")
        report_file.write(f"Test samples: {len(X_test):,}\n")
        report_file.write(f"Total features: {X.shape[1]:,}\n")
        report_file.write(f"  - Standard/Biological features: {n_standard}\n")
        report_file.write(f"  - FCGR (Fractal) features: {n_fcgr}\n")
        report_file.write(f"  - TDA (Topological) features: {n_tda}\n\n")

        report_file.write("Performance Metrics (Test Set)\n")
        for metric, value in metrics.items():
            report_file.write(f"{metric.upper():<30}: {value:.6f}\n")

        report_file.write("\n\nCalibration Analysis\n")
        report_file.write(f"Expected Calibration Error (ECE): {ece:.6f}\n")
        report_file.write(f"Brier Score: {brier:.6f}\n")

        report_file.write("\n\nCross-Validation Results (Mean ± 95% CI)\n")
        for metric_name, stats in cv_summary.items():
            report_file.write(f"{metric_name.upper():<30}: {stats['formatted']}\n")

        report_file.write("\n\nMcNemar's Significance Test (Enhanced vs Baseline)\n")
        report_file.write(f"McNemar Statistic: {mcnemar_result['statistic']:.4f}\n")
        report_file.write(f"P-value: {mcnemar_result['p_value']:.6f}\n")
        report_file.write(f"Significance: {mcnemar_result['significance']}\n")
        report_file.write(f"Interpretation: {mcnemar_result['interpretation']}\n")
        report_file.write(
            f"Enhanced Model Accuracy: {mcnemar_result['model1_accuracy']:.4f}\n"
        )
        report_file.write(
            f"Baseline Model Accuracy: {mcnemar_result['model2_accuracy']:.4f}\n"
        )
        report_file.write(
            f"Accuracy Improvement: {mcnemar_result['accuracy_difference']:+.4f}\n"
        )

        report_file.write("\n\nError Analysis Summary\n")
        n_fp = np.sum(misclass_df["error_type"] == "FP") if len(misclass_df) > 0 else 0
        n_fn = np.sum(misclass_df["error_type"] == "FN") if len(misclass_df) > 0 else 0
        report_file.write(f"Total Misclassifications: {len(misclass_df)}\n")
        report_file.write(f"False Positives (FP): {n_fp}\n")
        report_file.write(f"False Negatives (FN): {n_fn}\n")

        report_file.write("\n\nClassification Report\n")
        report_file.write(
            classification_report(
                y_test, y_pred, target_names=["Benign", "Pathogenic"], digits=4
            )
        )

        report_file.write("\n\nGenerated Output Files\n")
        report_file.write("Metrics & Reports:\n")
        report_file.write("  - final_metrics.csv\n")
        report_file.write("  - final_predictions.csv\n")
        report_file.write("  - final_report.txt\n")
        report_file.write("  - cv_metrics.csv\n")
        report_file.write("  - cv_summary_stats.csv\n")
        report_file.write("  - calibration_bins.csv\n")
        report_file.write("  - significance_test_mcnemar.csv\n")
        report_file.write("  - misclassifications.csv\n")
        report_file.write("  - feature_importance_aggregated.csv\n")
        report_file.write("  - learning_curve_data.csv\n")
        report_file.write("\nFigures:\n")
        report_file.write("  - Fig1_class_distribution.png/.tiff\n")
        report_file.write("  - Fig1_correlation_heatmap.png/.tiff\n")
        report_file.write("  - Fig1_ROC_PR_curves.png/.tiff\n")
        report_file.write("  - Fig_CM_TF-DFE.png/.tiff\n")
        report_file.write("  - Fig_CM_TF-DFE_normalized.png/.tiff\n")
        report_file.write("  - Fig_tSNE_TFDFE.png/.tiff\n")
        report_file.write("  - Fig_Calibration_Curve.png/.tiff\n")
        report_file.write("  - Fig_CV_Distribution.png/.tiff\n")
        report_file.write("  - Fig_Feature_Importance_Stacked.png/.tiff\n")
        report_file.write("  - Fig_Learning_Curve.png/.tiff\n")
        if SHAP_AVAILABLE:
            report_file.write("  - Fig_SHAP_summary.png/.tiff\n")
            report_file.write("  - Fig_SHAP_beeswarm.png/.tiff\n")

    print(f"  {Config.OUTPUT_DIR / 'final_report.txt'}")

    total_time = time.time() - start_time

    print(f"\nFINAL PUBLICATION RUN COMPLETE")

    print(f"\nFinal Performance (Test Set):")
    print(f"  MCC (Primary):      {metrics['mcc']:.6f}")
    print(f"  AUPRC:              {metrics['auprc']:.6f}")
    print(f"  AUROC:              {metrics['auroc']:.6f}")
    print(f"  F1-Score:           {metrics['f1']:.6f}")
    print(f"  ECE (Calibration):  {metrics.get('ece', ece):.6f}")

    print(f"\nCross-Validation (Mean ± 95% CI):")
    print(f"  MCC:   {cv_summary['mcc']['formatted']}")
    print(f"  AUPRC: {cv_summary['auprc']['formatted']}")

    print(f"\nStatistical Significance:")
    print(
        f"  McNemar p-value: {mcnemar_result['p_value']:.6f} ({mcnemar_result['significance']})"
    )
    print(f"  {mcnemar_result['interpretation']}")

    print(f"\nOutput Directory: {Config.OUTPUT_DIR}")

    print(f"\nGenerated CSV Files:")
    print(f"  - final_metrics.csv            (Primary performance metrics)")
    print(f"  - final_predictions.csv        (Test set predictions)")
    print(f"  - cv_metrics.csv               (Per-fold CV metrics)")
    print(f"  - cv_summary_stats.csv         (Mean ± 95% CI)")
    print(f"  - calibration_bins.csv         (Calibration bin statistics)")
    print(f"  - significance_test_mcnemar.csv (McNemar's test results)")
    print(f"  - misclassifications.csv       (Error analysis)")
    print(f"  - feature_importance_aggregated.csv (Category-level SHAP)")
    print(f"  - learning_curve_data.csv      (Performance vs data size)")

    print(f"\nGenerated Figures:")
    print(f"  [EDA]")
    print(f"    - Fig1_class_distribution.png/.tiff")
    print(f"    - Fig1_correlation_heatmap.png/.tiff")
    print(f"  [Performance]")
    print(f"    - Fig1_ROC_PR_curves.png/.tiff")
    print(f"    - Fig_CM_TF-DFE.png/.tiff (normalized & raw)")
    print(f"  [Calibration]")
    print(f"    - Fig_Calibration_Curve.png/.tiff")
    print(f"  [Statistical Robustness]")
    print(f"    - Fig_CV_Distribution.png/.tiff")
    print(f"    - Fig_Learning_Curve.png/.tiff")
    print(f"  [Feature Analysis]")
    print(f"    - Fig_Feature_Importance_Stacked.png/.tiff")
    print(f"    - Fig_tSNE_TFDFE.png/.tiff")
    if SHAP_AVAILABLE:
        print(f"    - Fig_SHAP_summary.png/.tiff")
        print(f"    - Fig_SHAP_beeswarm.png/.tiff")

    print(f"\nTotal Runtime: {total_time/60:.2f} minutes")

    print("\nReady for BMC Genomics submission!")


if __name__ == "__main__":
    run_final_model()
