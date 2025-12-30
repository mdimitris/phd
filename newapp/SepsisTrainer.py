import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score,  average_precision_score,  matthews_corrcoef
from LSTMImputer import LSTMModel
from GRUModel import GRUModel
import time
import lightgbm as lgb
import xgboost as xgb

class SepsisTrainer:
    def __init__(
        self,
        features,
        label_col="label_sepsis_within_6h",
        seq_len=30,
        hidden_size=64,
        num_layers=2,
        batch_size=32,
        model_type="lstm",
        dropout=0.2,
        lr=1e-3,
        epochs=10,
        early_stop_patience=2,         
        early_stop_metric="auprc", 
        mcc_threshold_grid=None,   
        tree_params=None, 
        early_stopping_rounds=50     
    ):
        self.features = features
        self.label_col = label_col
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model_type = model_type
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.tree_params = tree_params or {}
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stop_patience = early_stop_patience
        self.early_stop_metric = early_stop_metric.lower()

        # threshold candidates for MCC optimization
        if mcc_threshold_grid is None:
            self.mcc_threshold_grid = np.linspace(0.05, 0.95, 19)
        else:
            self.mcc_threshold_grid = np.array(mcc_threshold_grid, dtype=float)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SepsisTrainer using device: {self.device}")

        self.scaler_X = StandardScaler()
        self.model = None

        self.train_loader = None
        self.val_loader = None

        # NEW: store class weight for BCEWithLogitsLoss
        self.pos_weight = None

        # NEW: best checkpoint
        self.best_state = None
        self.best_score = -np.inf

    def _create_sequences_from_df(self, df: pd.DataFrame):
        X_seqs, y_seqs = [], []
        for _, g in df.groupby("stay_id"):
            values_X = g[self.features].values
            values_y = g[self.label_col].values

            if len(g) <= self.seq_len:
                continue

            for i in range(len(g) - self.seq_len):
                X_seqs.append(values_X[i : i + self.seq_len])
                y_seqs.append(values_y[i + self.seq_len])

        if not X_seqs:
            raise ValueError("No sequences were created. Check seq_len and data size.")

        X = np.array(X_seqs, dtype=np.float32)
        y = np.array(y_seqs, dtype=np.float32).reshape(-1, 1)
        return X, y
    
    def prepare_data_from_splits(self, df_train: pd.DataFrame, df_val: pd.DataFrame, time_col="charttime"):
        print("Preparing data from explicit train/val splits...")

        df_train = df_train.copy()
        df_val = df_val.copy()

        # Encode gender if needed
        for df in (df_train, df_val):
            if "gender" in df.columns and df["gender"].dtype == "object":
                df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype("float32")
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df[self.label_col] = df[self.label_col].astype("float32")

        # Drop rows missing essentials
        needed = ["stay_id", time_col, self.label_col] + self.features
        df_train = df_train.dropna(subset=[c for c in needed if c in df_train.columns]).copy()
        df_val = df_val.dropna(subset=[c for c in needed if c in df_val.columns]).copy()

        # Sort by stay + time
        df_train = df_train.sort_values(["stay_id", time_col]).reset_index(drop=True)
        df_val = df_val.sort_values(["stay_id", time_col]).reset_index(drop=True)

        # Fit scaler on TRAIN ONLY
        self.scaler_X.fit(df_train[self.features])
        df_train[self.features] = self.scaler_X.transform(df_train[self.features])
        df_val[self.features] = self.scaler_X.transform(df_val[self.features])

        # Create sequences
        X_train, y_train = self._create_sequences_from_df(df_train)
        X_val, y_val = self._create_sequences_from_df(df_val)
        print(f"Train sequences: {X_train.shape[0]}, Val sequences: {X_val.shape[0]}")

        # pos_weight from TRAIN sequences
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        if pos <= 0:
            raise ValueError("No positive samples in training data after sequencing.")
        pos_weight_value = neg / pos
        self.pos_weight = torch.tensor([pos_weight_value], device=self.device)
        print(f"pos_weight = {pos_weight_value:.2f} (neg={int(neg)}, pos={int(pos)})")

        # Dataloaders
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Model (re-init each fold)
        input_size = len(self.features)

        if self.model_type == "gru":
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
            ).to(self.device)
        else:
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
            ).to(self.device)

        print("Fold data preparation complete.")


    def prepare_data(self, df: pd.DataFrame, time_col="charttime"):
        print(f"Preparing data for sepsis {self.model_type.upper()} model...")


        df = df.copy()

        # Encode gender if needed
        if "gender" in df.columns and df["gender"].dtype == "object":
            df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype("float32")

        # Ensure datetime for sorting if time_col is charttime-like
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        # Ensure label numeric
        df[self.label_col] = df[self.label_col].astype("float32")

        # Drop rows missing essentials (avoid NaNs flowing into scaler/model)
        needed = ["stay_id", time_col, self.label_col] + self.features
        df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
        print(f"Rows (after cleaning): {len(df)}")
        print(f"Patients: {df['subject_id'].nunique()}")
        print(f"Stays: {df['stay_id'].nunique()}")
        # Sort by stay + time
        df = df.sort_values(["stay_id", time_col]).reset_index(drop=True)

        # Split by stay_id
        unique_stays = df["stay_id"].unique()
        train_stays, val_stays = train_test_split(
            unique_stays, test_size=0.3, random_state=42
        )

        df_train = df[df["stay_id"].isin(train_stays)].copy()
        df_val = df[df["stay_id"].isin(val_stays)].copy()
        print(f"Train stays: {len(train_stays)}, Val stays: {len(val_stays)}")

        # ✅ Fit scaler on TRAIN ONLY (avoids leakage)
        self.scaler_X.fit(df_train[self.features])
        df_train[self.features] = self.scaler_X.transform(df_train[self.features])
        df_val[self.features] = self.scaler_X.transform(df_val[self.features])

        # Create sequences
        X_train, y_train = self._create_sequences_from_df(df_train)
        X_val, y_val = self._create_sequences_from_df(df_val)
        

        print(f"Train sequences: {X_train.shape[0]}, Val sequences: {X_val.shape[0]}")

        # ✅ Compute pos_weight from TRAIN sequences (best practice)
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        if pos <= 0:
            raise ValueError("No positive samples in training data after sequencing.")
        pos_weight_value = neg / pos
        self.pos_weight = torch.tensor([pos_weight_value], device=self.device)
        print(f"pos_weight = {pos_weight_value:.2f} (neg={int(neg)}, pos={int(pos)})")

        # Tensors / loaders
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Model
        input_size = len(self.features)

        if self.model_type == "gru":
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
                dropout=self.dropout,
            ).to(self.device)
        else:
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1,
            ).to(self.device)

        print("Data preparation complete.")

    def train(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("Call prepare_data(df) before train().")

        # ✅ Use class weighting
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"Starting training for {self.epochs} epochs...")
        self.best_state = None
        self.best_score = -np.inf
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))

            # Evaluate
            val_loss, val_auc, val_acc, val_auprc, val_mcc, best_thr = self.evaluate(self.val_loader)

            print(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val AUPRC: {val_auprc:.4f} | "
                f"Val MCC: {val_mcc:.4f} (thr={best_thr:.2f}) | "
                f"Val Acc: {val_acc:.4f}"
            )

            # ✅ Early stopping on chosen metric
            if self.early_stop_metric == "auprc":
                score = val_auprc
            elif self.early_stop_metric == "mcc":
                score = val_mcc
            else:
                score = val_auc

            if score > self.best_score:
                self.best_score = score
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

        # ✅ Restore best checkpoint
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        print("Training complete (best checkpoint restored).")

    def evaluate(self, loader: DataLoader):
        self.model.eval()

        # Use the same weighting for loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        all_logits = []
        all_labels = []
        losses = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                losses.append(loss.item())

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(y_batch.detach().cpu().numpy())

        all_logits = np.vstack(all_logits).reshape(-1)   # (N,)
        all_labels = np.vstack(all_labels).reshape(-1)   # (N,)

        # probs in [0,1]
        probs = 1.0 / (1.0 + np.exp(-all_logits))

        # default threshold for accuracy
        preds_05 = (probs >= 0.5).astype(np.int32)

        # AUC / AUPRC
        try:
            auc = roc_auc_score(all_labels, probs)
        except ValueError:
            auc = float("nan")

        try:
            auprc = average_precision_score(all_labels, probs)
        except ValueError:
            auprc = float("nan")

        acc = accuracy_score(all_labels, preds_05)

        # ✅ Best-threshold MCC
        best_mcc = -1.0
        best_thr = 0.5
        for t in self.mcc_threshold_grid:
            preds_t = (probs >= t).astype(np.int32)
            mcc_t = matthews_corrcoef(all_labels, preds_t)
            if mcc_t > best_mcc:
                best_mcc = mcc_t
                best_thr = float(t)

        return float(np.mean(losses)), float(auc), float(acc), float(auprc), float(best_mcc), float(best_thr)
    

    #for xgboost and lightgbm
    def prepare_tabular_from_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ):
        """
        Prepare tabular (non-sequential) data for tree models (XGBoost / LightGBM).

        - Uses ONLY train split to compute class imbalance
        - Returns numpy-ready X / y
        - No time leakage
        """

        df_train = df_train.copy()
        df_val = df_val.copy()

        # ----------------------------
        # Basic preprocessing
        # ----------------------------
        for df in (df_train, df_val):
            if "gender" in df.columns and df["gender"].dtype == "object":
                df["gender"] = df["gender"].map({"M": 1, "F": 0}).astype("float32")

            # tree models want integer labels
            df[self.label_col] = df[self.label_col].astype(int)

        # ----------------------------
        # Drop rows with missing values
        # ----------------------------
        needed = [self.label_col] + self.features
        df_train = df_train.dropna(subset=needed).copy()
        df_val = df_val.dropna(subset=needed).copy()

        # ----------------------------
        # Split X / y
        # ----------------------------
        X_train = df_train[self.features].values
        y_train = df_train[self.label_col].values

        X_val = df_val[self.features].values
        y_val = df_val[self.label_col].values

        # ----------------------------
        # Class imbalance (TRAIN ONLY)
        # ----------------------------
        pos = y_train.sum()
        neg = len(y_train) - pos

        if pos == 0:
            raise ValueError("No positive samples in training split.")

        self.scale_pos_weight = float(neg / pos)

        return X_train, y_train, X_val, y_val




    def train_tree_from_splits(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        X_train, y_train, X_val, y_val = self.prepare_tabular_from_splits(df_train, df_val)

        t0 = time.time()

        if self.model_type == "lgbm":
            params = dict(
                n_estimators=5000,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary",
                n_jobs=-1,
                random_state=42,
            )
            params.update(self.tree_params)

            model = lgb.LGBMClassifier(**params, scale_pos_weight=self.scale_pos_weight)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            probs = model.predict_proba(X_val)[:, 1]
            best_iter = getattr(model, "best_iteration_", None)

        elif self.model_type == "xgb":

            # 1️⃣ Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval   = xgb.DMatrix(X_val, label=y_val)

            # 2️⃣ XGBoost native params (IMPORTANT differences marked)
            params = dict(
                max_depth=4,
                eta=0.03,                     # ← learning_rate → eta
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,                  # ← reg_lambda → lambda_
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                scale_pos_weight=self.scale_pos_weight,
                seed=42,
            )

            params.update(self.tree_params)

            # 3️⃣ Train with early stopping (THIS NOW WORKS)
            t0 = time.time()

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=2000,         # ← n_estimators → num_boost_round
                evals=[(dval, "val")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )

            time_min = (time.time() - t0) / 60.0

            # 4️⃣ Outputs
            best_iter = model.best_iteration
            probs = model.predict(dval)

            #best_iter = getattr(model, "best_iteration", None)

        else:
            raise ValueError(f"Unsupported tree model_type: {self.model_type}")

        time_min = (time.time() - t0) / 60.0

        # reuse your existing evaluate logic (but it expects a loader)
        # so do a small inline evaluation here:
        auc = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else float("nan")
        auprc = average_precision_score(y_val, probs) if len(np.unique(y_val)) > 1 else float("nan")
        preds_05 = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds_05)

        best_mcc, best_thr = -1.0, 0.5
        for t in self.mcc_threshold_grid:
            preds_t = (probs >= t).astype(int)
            mcc_t = matthews_corrcoef(y_val, preds_t)
            if mcc_t > best_mcc:
                best_mcc, best_thr = mcc_t, float(t)

        return dict(
            auc=float(auc),
            auprc=float(auprc),
            acc=float(acc),
            mcc=float(best_mcc),
            thr=float(best_thr),
            time_min=float(time_min),
            best_iter=best_iter,
            scale_pos_weight=float(self.scale_pos_weight),
        )



